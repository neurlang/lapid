//! Temp memory planning: linear-scan slot assignment over the launch order.
//!
//! A temp is live from the first launch touching it (its producer) to the last launch
//! touching it. Two temps may share a slot only if their live ranges do not overlap — a
//! launch that reads one temp never writes another temp of the same slot. Free slots are
//! reused best-fit (smallest slot that is big enough, else the largest one, grown).

use crate::ir::{BufferSource, BufferSpec, Launch, OutputSpec, Schedule};

use super::{Buf, BufKind};

pub(crate) fn plan(bufs: Vec<Buf>, launches: Vec<Launch>, outputs: Vec<OutputSpec>) -> Schedule {
    let n = bufs.len();
    let mut first = vec![usize::MAX; n];
    let mut last = vec![0usize; n];
    for (i, l) in launches.iter().enumerate() {
        for &b in l.inputs.iter().chain(std::iter::once(&l.output)) {
            if b < n {
                if first[b] == usize::MAX {
                    first[b] = i;
                }
                last[b] = i;
            }
        }
    }
    let mut temps: Vec<usize> = (0..n).filter(|&b| matches!(bufs[b].kind, BufKind::Temp)).collect();
    for &b in &temps {
        if first[b] == usize::MAX {
            // Never touched (e.g. a zero-sized value): give it a trivial range.
            first[b] = 0;
            last[b] = 0;
        }
    }
    temps.sort_by_key(|&b| (first[b], b));

    let mut slot_bytes: Vec<usize> = Vec::new();
    let mut busy_until: Vec<usize> = Vec::new();
    let mut slot_of = vec![usize::MAX; n];
    for &b in &temps {
        let (start, need) = (first[b], bufs[b].bytes);
        let mut best: Option<usize> = None;
        for s in 0..slot_bytes.len() {
            if busy_until[s] >= start {
                continue;
            }
            best = match best {
                None => Some(s),
                Some(c) if better(slot_bytes[c], slot_bytes[s], need) => Some(s),
                keep => keep,
            };
        }
        let s = match best {
            Some(s) => {
                slot_bytes[s] = slot_bytes[s].max(need);
                busy_until[s] = last[b];
                s
            }
            None => {
                slot_bytes.push(need);
                busy_until.push(last[b]);
                slot_bytes.len() - 1
            }
        };
        slot_of[b] = s;
    }

    let buffers = bufs
        .into_iter()
        .enumerate()
        .map(|(i, b)| BufferSpec {
            bytes: b.bytes,
            source: match b.kind {
                BufKind::Host(h) => BufferSource::Host(h),
                BufKind::Device(d) => BufferSource::Device(d),
                BufKind::Temp => BufferSource::Temp { slot: slot_of[i] },
                BufKind::Output => BufferSource::Output,
            },
        })
        .collect();
    Schedule { buffers, slot_bytes, launches, outputs }
}

/// Is a free slot of size `cand` a better home than one of size `cur` for `need` bytes?
fn better(cur: usize, cand: usize, need: usize) -> bool {
    match (cur >= need, cand >= need) {
        (true, true) => cand < cur,
        (false, true) => true,
        (true, false) => false,
        (false, false) => cand > cur,
    }
}
