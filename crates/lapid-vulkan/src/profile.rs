//! GPU timestamp profiling: one timestamp before the first dispatch and one after every
//! dispatch of a command buffer, so launch `i` costs `t[i + 1] - t[i]` (including any
//! barrier wait in front of it). Enabled with `LAPID_PROFILE=1` (totals per kernel kind,
//! printed when the backend is dropped) and used by `VulkanBackend::time_launches`.

use std::collections::HashMap;

use ash::vk;
use lapid_tensor::Result;

use crate::context::{Shared, vk_err};

pub(crate) struct Profiler {
    pool: vk::QueryPool,
    cap: u32,
    period_ns: f64,
    /// label → (launches, total ms)
    totals: HashMap<String, (u64, f64)>,
    /// Host time spent planning + recording + submitting (ms) and number of executes.
    pub host_ms: f64,
    pub executes: u64,
}

impl Profiler {
    pub fn new(shared: &Shared) -> Option<Profiler> {
        let period = shared.caps.timestamp_period;
        if period <= 0.0 {
            return None;
        }
        Some(Profiler {
            pool: vk::QueryPool::null(),
            cap: 0,
            period_ns: period as f64,
            totals: HashMap::new(),
            host_ms: 0.0,
            executes: 0,
        })
    }

    /// Make room for `n` timestamps and record the pool reset into `cmd`.
    pub fn begin(&mut self, shared: &Shared, cmd: vk::CommandBuffer, n: u32) -> Result<()> {
        let dev = &shared.device;
        if n > self.cap {
            let cap = n.max(1024).next_power_of_two();
            // SAFETY: the old pool is idle (previous submission completed).
            unsafe {
                if self.pool != vk::QueryPool::null() {
                    dev.destroy_query_pool(self.pool, None);
                    self.pool = vk::QueryPool::null();
                    self.cap = 0;
                }
                let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(cap);
                self.pool = dev.create_query_pool(&info, None).map_err(|e| vk_err("vkCreateQueryPool", e))?;
            }
            self.cap = cap;
        }
        // SAFETY: recording state, pool sized for `n` queries.
        unsafe { dev.cmd_reset_query_pool(cmd, self.pool, 0, n) };
        Ok(())
    }

    /// Record timestamp `i` (after all previously recorded work completes).
    pub fn stamp(&self, dev: &ash::Device, cmd: vk::CommandBuffer, i: u32) {
        // SAFETY: recording state; `i` < the count passed to `begin`.
        unsafe { dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, i) };
    }

    /// Durations (ms) between consecutive timestamps `0..n` of the completed submission.
    pub fn read(&self, shared: &Shared, n: u32) -> Result<Vec<f64>> {
        let mut ticks = vec![0u64; n as usize];
        // SAFETY: the submission writing these queries has completed (fence waited).
        unsafe {
            shared
                .device
                .get_query_pool_results(self.pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .map_err(|e| vk_err("vkGetQueryPoolResults", e))?;
        }
        Ok(ticks.windows(2).map(|w| w[1].wrapping_sub(w[0]) as f64 * self.period_ns * 1e-6).collect())
    }

    pub fn add(&mut self, label: &str, ms: f64) {
        let e = self.totals.entry(label.to_string()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }

    pub fn reset(&mut self) {
        self.totals.clear();
        self.host_ms = 0.0;
        self.executes = 0;
    }

    /// Table of totals, most expensive first.
    pub fn report(&self) -> String {
        let mut rows: Vec<(&String, &(u64, f64))> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let total: f64 = rows.iter().map(|r| r.1.1).sum();
        let mut s = format!("{:<28} {:>8} {:>11} {:>10} {:>6}\n", "kernel", "launches", "total ms", "avg us", "%");
        for (label, &(n, ms)) in rows {
            s.push_str(&format!(
                "{label:<28} {n:>8} {ms:>11.3} {:>10.2} {:>5.1}%\n",
                ms * 1e3 / n.max(1) as f64,
                100.0 * ms / total.max(1e-12)
            ));
        }
        s.push_str(&format!(
            "{:<28} {:>8} {total:>11.3}   (host record+submit {:.3} ms over {} executes)\n",
            "gpu total", "", self.host_ms, self.executes
        ));
        s
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        if self.pool != vk::QueryPool::null() {
            // SAFETY: device idle at teardown.
            unsafe { device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}
