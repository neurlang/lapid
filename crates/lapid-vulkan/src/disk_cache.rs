//! On-disk kernel cache: the SPIR-V of every compiled kernel plus the driver's
//! `VkPipelineCache` blob, under `<lapid cache>/kernels/vulkan/<device key>/`.
//!
//! The lapid cache dir is `$LAPID_CACHE`, else `$XDG_CACHE_HOME/lapid`, else `~/.cache/lapid`.
//! The device key covers vendor / device id, driver version, the pipeline cache UUID and the
//! lapid version; SPIR-V files are named by a 128-bit hash of the final WGSL plus the device
//! capabilities used to compile it. Every read is validated (magic, length, checksum,
//! pipeline-cache header) and anything unexpected is ignored, never an error. Writes go
//! through a temp file + rename. `LAPID_NO_KERNEL_CACHE=1` disables the cache.

use std::path::{Path, PathBuf};

/// Identity of the device + driver the cached binaries were built for.
#[derive(Clone, Debug)]
pub(crate) struct DeviceId {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub pipeline_cache_uuid: [u8; 16],
}

pub(crate) struct DiskCache {
    dir: PathBuf,
    id: DeviceId,
}

const SPV_MAGIC: &[u8; 4] = b"GSPV";

/// FNV-1a 64 with a caller-chosen offset basis.
fn fnv(bytes: &[u8], basis: u64) -> u64 {
    let mut h = basis;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 128-bit hex hash (two independent FNV-1a streams) of `parts`.
pub(crate) fn hash_hex(parts: &[&[u8]]) -> String {
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64, 0x6c62_272e_07bb_0142u64);
    for p in parts {
        a = fnv(p, a);
        b = fnv(p, b ^ 0x9e37_79b9_7f4a_7c15);
        // separator so ("ab", "c") != ("a", "bc")
        a = fnv(&[0xff], a);
        b = fnv(&[0xfe], b);
    }
    format!("{a:016x}{b:016x}")
}

fn nonempty(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The lapid cache dir (same rule as `lapid-convert::cache_dir`).
fn cache_root() -> Option<PathBuf> {
    if let Some(p) = nonempty("LAPID_CACHE") {
        return Some(p);
    }
    if let Some(p) = nonempty("XDG_CACHE_HOME") {
        return Some(p.join("lapid"));
    }
    nonempty("HOME").map(|h| h.join(".cache").join("lapid"))
}

fn write_atomic(path: &Path, data: &[u8]) {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp{}-{n}", std::process::id()));
    if std::fs::write(&tmp, data).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

impl DiskCache {
    /// `None` when disabled (`LAPID_NO_KERNEL_CACHE=1`) or no cache dir can be determined.
    pub fn open(id: &DeviceId) -> Option<DiskCache> {
        if std::env::var("LAPID_NO_KERNEL_CACHE").is_ok_and(|v| !v.is_empty() && v != "0") {
            return None;
        }
        let key = hash_hex(&[
            &id.vendor_id.to_le_bytes(),
            &id.device_id.to_le_bytes(),
            &id.driver_version.to_le_bytes(),
            &id.pipeline_cache_uuid,
            env!("CARGO_PKG_VERSION").as_bytes(),
            id.name.as_bytes(),
        ]);
        let slug: String = id.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).take(32).collect();
        let dir = cache_root()?.join("kernels").join("vulkan").join(format!("{slug}-{}", &key[..16]));
        Some(DiskCache { dir, id: id.clone() })
    }

    fn ensure_dir(&self) -> bool {
        std::fs::create_dir_all(&self.dir).is_ok()
    }

    /// Cached SPIR-V for `hash`, if present and intact.
    pub fn load_spirv(&self, hash: &str) -> Option<Vec<u32>> {
        let bytes = std::fs::read(self.dir.join(format!("{hash}.spv"))).ok()?;
        if bytes.len() < 16 || &bytes[..4] != SPV_MAGIC {
            return None;
        }
        let n = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let sum = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let body = &bytes[16..];
        if body.len() != n * 4 || n == 0 || fnv(body, 0xcbf2_9ce4_8422_2325) != sum {
            return None;
        }
        let words: Vec<u32> = body.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        (words[0] == 0x0723_0203).then_some(words)
    }

    pub fn store_spirv(&self, hash: &str, words: &[u32]) {
        if !self.ensure_dir() {
            return;
        }
        let body: &[u8] = bytemuck::cast_slice(words);
        let mut out = Vec::with_capacity(16 + body.len());
        out.extend_from_slice(SPV_MAGIC);
        out.extend_from_slice(&(words.len() as u32).to_le_bytes());
        out.extend_from_slice(&fnv(body, 0xcbf2_9ce4_8422_2325).to_le_bytes());
        out.extend_from_slice(body);
        write_atomic(&self.dir.join(format!("{hash}.spv")), &out);
    }

    fn pipeline_cache_path(&self) -> PathBuf {
        self.dir.join("pipeline-cache.bin")
    }

    /// The saved `VkPipelineCache` data if its header matches this device.
    pub fn load_pipeline_cache(&self) -> Option<Vec<u8>> {
        let data = std::fs::read(self.pipeline_cache_path()).ok()?;
        self.header_ok(&data).then_some(data)
    }

    /// VkPipelineCacheHeaderVersionOne: length, version 1, vendor, device, uuid.
    fn header_ok(&self, d: &[u8]) -> bool {
        let word = |i: usize| d.get(i..i + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        d.len() >= 32
            && word(0).is_some_and(|l| l >= 32 && l as usize <= d.len())
            && word(4) == Some(1)
            && word(8) == Some(self.id.vendor_id)
            && word(12) == Some(self.id.device_id)
            && d[16..32] == self.id.pipeline_cache_uuid
    }

    pub fn store_pipeline_cache(&self, data: &[u8]) {
        if self.header_ok(data) && self.ensure_dir() {
            write_atomic(&self.pipeline_cache_path(), data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> DeviceId {
        DeviceId { name: "Test GPU".into(), vendor_id: 0x10de, device_id: 0x1234, driver_version: 7, pipeline_cache_uuid: [3; 16] }
    }

    #[test]
    fn spirv_roundtrip_and_corruption() {
        let dir = std::env::temp_dir().join(format!("lapid-vk-cache-test-{}", std::process::id()));
        let c = DiskCache { dir: dir.clone(), id: id() };
        let words = vec![0x0723_0203u32, 1, 2, 3];
        c.store_spirv("abc", &words);
        assert_eq!(c.load_spirv("abc"), Some(words));
        // corrupt one byte -> ignored
        let p = dir.join("abc.spv");
        let mut b = std::fs::read(&p).unwrap();
        let last = b.len() - 1;
        b[last] ^= 1;
        std::fs::write(&p, &b).unwrap();
        assert_eq!(c.load_spirv("abc"), None);
        std::fs::write(&p, b"junk").unwrap();
        assert_eq!(c.load_spirv("abc"), None);
        assert_eq!(c.load_spirv("missing"), None);
        // pipeline cache header must match the device
        let mut blob = vec![0u8; 40];
        blob[..4].copy_from_slice(&32u32.to_le_bytes());
        blob[4..8].copy_from_slice(&1u32.to_le_bytes());
        blob[8..12].copy_from_slice(&0x10deu32.to_le_bytes());
        blob[12..16].copy_from_slice(&0x1234u32.to_le_bytes());
        blob[16..32].copy_from_slice(&[3; 16]);
        c.store_pipeline_cache(&blob);
        assert_eq!(c.load_pipeline_cache(), Some(blob.clone()));
        blob[20] = 9;
        std::fs::write(c.pipeline_cache_path(), &blob).unwrap();
        assert_eq!(c.load_pipeline_cache(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_is_stable_and_separated() {
        assert_eq!(hash_hex(&[b"ab", b"c"]), hash_hex(&[b"ab", b"c"]));
        assert_ne!(hash_hex(&[b"ab", b"c"]), hash_hex(&[b"a", b"bc"]));
        assert_eq!(hash_hex(&[b"x"]).len(), 32);
    }
}
