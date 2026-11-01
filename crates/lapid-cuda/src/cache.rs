//! On-disk kernel cache: the driver JIT output (cubin from `cuLinkComplete`) per kernel,
//! under `$LAPID_CACHE` → `$XDG_CACHE_HOME/lapid` → `~/.cache/lapid`, in `kernels/cuda/`.
//!
//! The file name hashes the device (name, compute capability), the driver version, the
//! lapid version and the full PTX source (header included), so a stale entry is never
//! looked up. Entries carry a magic, the key and the payload length; anything that does not
//! match (or a cubin the driver rejects) is ignored and rewritten. The cache never fails a
//! compile: I/O errors only cost the cached path. Writes go to a temp file then `rename`
//! (atomic, safe with concurrent processes). `LAPID_NO_KERNEL_CACHE=1` disables it.
//!
//! This complements the driver's own JIT cache (`~/.nv/ComputeCache`, `CUDA_CACHE_*`),
//! which is shared with every CUDA program on the machine and evicts at its size limit;
//! loading a cubin also skips the driver's PTX parsing and cache lookup.

use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"LPIDCUB1";
const HEADER: usize = 24;

/// FNV-1a 64 (stable across Rust versions, unlike `DefaultHasher`).
pub(crate) fn fnv(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        // separator so ("ab", "c") != ("a", "bc")
        h ^= 0xff;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub(crate) struct DiskCache {
    dir: Option<PathBuf>,
    /// Device + driver + lapid version.
    salt: Vec<u8>,
}

fn enabled() -> bool {
    !std::env::var("LAPID_NO_KERNEL_CACHE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `$LAPID_CACHE` → `$XDG_CACHE_HOME/lapid` → `~/.cache/lapid` (+ `kernels/cuda`).
fn cache_dir() -> Option<PathBuf> {
    let nonempty = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = nonempty("LAPID_CACHE")
        .or_else(|| nonempty("XDG_CACHE_HOME").map(|p| p.join("lapid")))
        .or_else(|| nonempty("HOME").map(|p| p.join(".cache").join("lapid")))?;
    Some(base.join("kernels").join("cuda"))
}

impl DiskCache {
    pub fn new(device: &str, sm: u32, driver: (u32, u32)) -> DiskCache {
        let salt = format!(
            "{device}|sm_{sm}|driver {}.{}|lapid {}",
            driver.0,
            driver.1,
            env!("CARGO_PKG_VERSION")
        );
        DiskCache {
            dir: if enabled() { cache_dir() } else { None },
            salt: salt.into_bytes(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    fn key(&self, src: &str) -> u64 {
        fnv(&[&self.salt, src.as_bytes()])
    }

    fn path(&self, key: u64) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{key:016x}.cubin")))
    }

    /// The cached cubin of `src`, if present and intact.
    pub fn get(&self, src: &str) -> Option<Vec<u8>> {
        let key = self.key(src);
        let data = std::fs::read(self.path(key)?).ok()?;
        parse(&data, key).map(<[u8]>::to_vec)
    }

    /// Store the cubin of `src` (best effort).
    pub fn put(&self, src: &str, cubin: &[u8]) {
        let key = self.key(src);
        let (Some(dir), Some(path)) = (self.dir.as_ref(), self.path(key)) else {
            return;
        };
        let mut data = Vec::with_capacity(HEADER + cubin.len());
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&(cubin.len() as u64).to_le_bytes());
        data.extend_from_slice(cubin);
        let _ = write_atomic(dir, &path, &data);
    }
}

fn parse(data: &[u8], key: u64) -> Option<&[u8]> {
    if data.len() < HEADER || &data[..8] != MAGIC {
        return None;
    }
    let k = u64::from_le_bytes(data[8..16].try_into().ok()?);
    let n = u64::from_le_bytes(data[16..24].try_into().ok()?);
    let body = &data[HEADER..];
    (k == key && n == body.len() as u64 && !body.is_empty()).then_some(body)
}

fn write_atomic(dir: &Path, path: &Path, data: &[u8]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".tmp-{}-{:x}",
        std::process::id(),
        fnv(&[path.as_os_str().as_encoded_bytes()])
    ));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_validate() {
        let dir = std::env::temp_dir().join(format!("lapid-cuda-cache-test-{}", std::process::id()));
        let c = DiskCache {
            dir: Some(dir.clone()),
            salt: b"dev".to_vec(),
        };
        assert!(c.get("ptx a").is_none());
        c.put("ptx a", b"cubin bytes");
        assert_eq!(c.get("ptx a").as_deref(), Some(&b"cubin bytes"[..]));
        assert!(c.get("ptx b").is_none());
        // corrupt (truncated) entries are ignored
        let p = c.path(c.key("ptx a")).unwrap();
        let d = std::fs::read(&p).unwrap();
        std::fs::write(&p, &d[..d.len() - 1]).unwrap();
        assert!(c.get("ptx a").is_none());
        std::fs::write(&p, b"garbage").unwrap();
        assert!(c.get("ptx a").is_none());
        // another salt (device / driver / version) never sees the entry
        c.put("ptx a", b"cubin bytes");
        let other = DiskCache {
            dir: Some(dir.clone()),
            salt: b"other".to_vec(),
        };
        assert!(other.get("ptx a").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
