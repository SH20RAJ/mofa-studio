//! Background system monitor for CPU, memory, and GPU usage
//!
//! This module provides a thread-safe system monitor that polls CPU, memory,
//! and GPU usage in a background thread, keeping the UI thread free.
//!
//! GPU monitoring:
//! - macOS: Uses IOKit via `ioreg` command to query GPU statistics
//! - Linux/Windows with NVIDIA: Uses nvml-wrapper (commented out, enable if needed)

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;
use sysinfo::System;

const USAGE_SCALE_MAX: u32 = 10_000;

/// One read-consistent snapshot of monitor values.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub vram_usage: f64,
    pub gpu_available: bool,
}

/// Shared system stats, updated by background thread
struct SystemStats {
    /// CPU usage scaled to 0-10000 (representing 0.00% to 100.00%)
    cpu_usage: AtomicU32,
    /// Memory usage scaled to 0-10000 (representing 0.00% to 100.00%)
    memory_usage: AtomicU32,
    /// GPU utilization scaled to 0-10000 (representing 0.00% to 100.00%)
    gpu_usage: AtomicU32,
    /// VRAM usage scaled to 0-10000 (representing 0.00% to 100.00%)
    vram_usage: AtomicU32,
    /// Whether GPU monitoring is available
    gpu_available: AtomicBool,
}

impl SystemStats {
    fn new() -> Self {
        Self {
            cpu_usage: AtomicU32::new(0),
            memory_usage: AtomicU32::new(0),
            gpu_usage: AtomicU32::new(0),
            vram_usage: AtomicU32::new(0),
            gpu_available: AtomicBool::new(false),
        }
    }
}

#[inline]
fn scale_percentage_to_u32(pct: f64) -> u32 {
    if !pct.is_finite() {
        return 0;
    }
    pct.clamp(0.0, 100.0).mul_add(100.0, 0.0).round() as u32
}

#[inline]
fn normalize_scaled_usage(value: u32) -> f64 {
    value as f64 / USAGE_SCALE_MAX as f64
}

/// Global system monitor instance
static SYSTEM_MONITOR: OnceLock<Arc<SystemStats>> = OnceLock::new();

// ============================================================================
// macOS GPU monitoring using IOKit via ioreg command
// ============================================================================

#[cfg(target_os = "macos")]
mod macos_gpu {
    use std::process::Command;

    /// GPU statistics from macOS IOKit
    #[derive(Debug, Default)]
    pub struct MacOSGpuStats {
        pub gpu_utilization: Option<f64>, // 0.0 - 1.0
        pub vram_used_mb: Option<u64>,
        pub vram_total_mb: Option<u64>,
    }

    /// Query GPU statistics from macOS using ioreg
    /// Parses IOAccelerator data for GPU utilization and VRAM info
    ///
    /// Apple Silicon format (M1/M2/M3):
    /// "PerformanceStatistics" = {"Device Utilization %"=0,"In use system memory"=1732460544,...}
    pub fn query_gpu_stats() -> MacOSGpuStats {
        let mut stats = MacOSGpuStats::default();

        // Query IOAccelerator for GPU statistics
        let output = Command::new("ioreg")
            .args(["-r", "-d", "1", "-c", "IOAccelerator"])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Find PerformanceStatistics line and parse values from dictionary
                for line in stdout.lines() {
                    let line = line.trim();

                    // Parse PerformanceStatistics dictionary
                    // Format: "PerformanceStatistics" = {"Device Utilization %"=0,"In use system memory"=123,...}
                    if line.contains("PerformanceStatistics") && line.contains("{") {
                        // Extract Device Utilization %
                        if let Some(util) = extract_dict_value(line, "Device Utilization %") {
                            stats.gpu_utilization = Some(util / 100.0);
                        }

                        // Extract memory stats for Apple Silicon (unified memory)
                        // "In use system memory" is GPU memory currently in use (bytes)
                        // "Alloc system memory" is total allocated for GPU (bytes)
                        if let Some(used_bytes) = extract_dict_value(line, "In use system memory\"")
                        {
                            stats.vram_used_mb = Some((used_bytes as u64) / (1024 * 1024));
                        }
                        if let Some(alloc_bytes) = extract_dict_value(line, "Alloc system memory") {
                            stats.vram_total_mb = Some((alloc_bytes as u64) / (1024 * 1024));
                        }

                        // Found what we need, stop searching
                        if stats.gpu_utilization.is_some() {
                            break;
                        }
                    }

                    // Fallback: discrete GPU format (Intel Macs with AMD/NVIDIA)
                    // VRAM Total (in MB)
                    if line.contains("VRAM,totalMB") {
                        if let Some(value) = extract_simple_value(line) {
                            stats.vram_total_mb = Some(value as u64);
                        }
                    }
                    // VRAM Free (calculate used from total - free)
                    if line.contains("VRAM,freeMB") {
                        if let Some(value) = extract_simple_value(line) {
                            if let Some(total) = stats.vram_total_mb {
                                stats.vram_used_mb = Some(total.saturating_sub(value as u64));
                            }
                        }
                    }
                }
            }
        }

        // Fallback for VRAM if not found
        if stats.vram_total_mb.is_none() {
            // Apple Silicon uses unified memory, estimate GPU portion
            let sys = sysinfo::System::new_all();
            let total_mb = sys.total_memory() / (1024 * 1024);
            // Apple Silicon can use up to ~75% of system memory for GPU
            stats.vram_total_mb = Some((total_mb * 3 / 4) as u64);
            stats.vram_used_mb = Some(0);
        }

        stats
    }

    /// Extract a value from dictionary format: "Key"=123 or "Key"=123,
    fn extract_dict_value(line: &str, key: &str) -> Option<f64> {
        // Find the key in the line
        let key_pos = line.find(key)?;
        let after_key = &line[key_pos + key.len()..];

        // Find the = sign after the key
        let eq_pos = after_key.find('=')?;
        let after_eq = &after_key[eq_pos + 1..];

        // Extract the number (stop at comma, space, or closing brace)
        let value_str: String = after_eq
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();

        value_str.parse().ok()
    }

    /// Extract a simple value from format: "key" = 123
    fn extract_simple_value(line: &str) -> Option<f64> {
        if let Some(eq_pos) = line.rfind('=') {
            let value_part = line[eq_pos + 1..].trim();
            let clean_value: String = value_part
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            clean_value.parse().ok()
        } else {
            None
        }
    }
}

/// Start the background system monitor thread if not already running.
/// This should be called once at app startup.
pub fn start_system_monitor() {
    SYSTEM_MONITOR.get_or_init(|| {
        let stats = Arc::new(SystemStats::new());
        let stats_clone = Arc::clone(&stats);

        thread::Builder::new()
            .name("system-monitor".to_string())
            .spawn(move || {
                let mut sys = System::new_all();

                // ============================================================
                // macOS GPU monitoring initialization
                // ============================================================
                #[cfg(target_os = "macos")]
                {
                    // Test if we can query GPU stats
                    let test_stats = macos_gpu::query_gpu_stats();
                    if test_stats.gpu_utilization.is_some() || test_stats.vram_total_mb.is_some() {
                        stats_clone.gpu_available.store(true, Ordering::Relaxed);
                        log::info!("GPU monitoring enabled (macOS IOKit)");
                    } else {
                        // Even if we can't get utilization, mark as available for VRAM display
                        stats_clone.gpu_available.store(true, Ordering::Relaxed);
                        log::info!("GPU monitoring enabled (macOS - limited stats available)");
                    }
                }

                #[cfg(not(target_os = "macos"))]
                {
                    log::info!("GPU monitoring not available (NVIDIA support commented out, enable in system_monitor.rs)");
                }

                loop {
                    // Refresh CPU and memory
                    sys.refresh_cpu_usage();
                    sys.refresh_memory();

                    // Get CPU usage (0.0 - 100.0)
                    let cpu = sys.global_cpu_usage();
                    let cpu_scaled = scale_percentage_to_u32(cpu);
                    stats_clone.cpu_usage.store(cpu_scaled, Ordering::Relaxed);

                    // Get memory usage
                    let total_memory = sys.total_memory();
                    let used_memory = sys.used_memory();
                    let memory_pct = if total_memory > 0 {
                        scale_percentage_to_u32(used_memory as f64 / total_memory as f64 * 100.0)
                    } else {
                        0
                    };
                    stats_clone.memory_usage.store(memory_pct, Ordering::Relaxed);

                    // ========================================================
                    // macOS GPU monitoring
                    // ========================================================
                    #[cfg(target_os = "macos")]
                    {
                        let gpu_stats = macos_gpu::query_gpu_stats();

                        // GPU utilization
                        if let Some(util) = gpu_stats.gpu_utilization {
                            let gpu_pct = scale_percentage_to_u32(util * 100.0);
                            stats_clone.gpu_usage.store(gpu_pct, Ordering::Relaxed);
                        }

                        // VRAM usage
                        if let (Some(used), Some(total)) = (gpu_stats.vram_used_mb, gpu_stats.vram_total_mb) {
                            if total > 0 {
                                let vram_pct = scale_percentage_to_u32((used as f64 / total as f64) * 100.0);
                                stats_clone.vram_usage.store(vram_pct, Ordering::Relaxed);
                            }
                        }
                    }

                    // Sleep for 1 second
                    thread::sleep(Duration::from_secs(1));
                }
            })
            .expect("Failed to spawn system monitor thread");

        stats
    });
}

/// Get current CPU usage as a value between 0.0 and 1.0
pub fn get_cpu_usage() -> f64 {
    SYSTEM_MONITOR
        .get()
        .map(|stats| normalize_scaled_usage(stats.cpu_usage.load(Ordering::Relaxed)))
        .unwrap_or(0.0)
}

/// Get current memory usage as a value between 0.0 and 1.0
pub fn get_memory_usage() -> f64 {
    SYSTEM_MONITOR
        .get()
        .map(|stats| normalize_scaled_usage(stats.memory_usage.load(Ordering::Relaxed)))
        .unwrap_or(0.0)
}

/// Get current GPU utilization as a value between 0.0 and 1.0
/// Returns 0.0 if GPU monitoring is not available
pub fn get_gpu_usage() -> f64 {
    SYSTEM_MONITOR
        .get()
        .map(|stats| normalize_scaled_usage(stats.gpu_usage.load(Ordering::Relaxed)))
        .unwrap_or(0.0)
}

/// Get current VRAM usage as a value between 0.0 and 1.0
/// Returns 0.0 if GPU monitoring is not available
pub fn get_vram_usage() -> f64 {
    SYSTEM_MONITOR
        .get()
        .map(|stats| normalize_scaled_usage(stats.vram_usage.load(Ordering::Relaxed)))
        .unwrap_or(0.0)
}

/// Check if GPU monitoring is available
pub fn is_gpu_available() -> bool {
    SYSTEM_MONITOR
        .get()
        .map(|stats| stats.gpu_available.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// Get a single snapshot of all system monitor values.
///
/// This avoids mixed reads across different monitor ticks in UI code.
pub fn get_system_snapshot() -> SystemSnapshot {
    SYSTEM_MONITOR
        .get()
        .map(|stats| SystemSnapshot {
            cpu_usage: normalize_scaled_usage(stats.cpu_usage.load(Ordering::Relaxed)),
            memory_usage: normalize_scaled_usage(stats.memory_usage.load(Ordering::Relaxed)),
            gpu_usage: normalize_scaled_usage(stats.gpu_usage.load(Ordering::Relaxed)),
            vram_usage: normalize_scaled_usage(stats.vram_usage.load(Ordering::Relaxed)),
            gpu_available: stats.gpu_available.load(Ordering::Relaxed),
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{normalize_scaled_usage, scale_percentage_to_u32};

    #[test]
    fn scale_percentage_clamps_invalid_values() {
        assert_eq!(scale_percentage_to_u32(f64::NAN), 0);
        assert_eq!(scale_percentage_to_u32(-12.0), 0);
        assert_eq!(scale_percentage_to_u32(120.0), 10_000);
    }

    #[test]
    fn normalize_scaled_usage_works() {
        assert_eq!(normalize_scaled_usage(0), 0.0);
        assert_eq!(normalize_scaled_usage(5_000), 0.5);
        assert_eq!(normalize_scaled_usage(10_000), 1.0);
    }
}
