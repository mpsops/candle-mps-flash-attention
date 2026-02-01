//! Test and benchmark flash attention

use candle_core::{Device, Tensor};
use candle_mps_flash_attention::flash_attention;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    println!("=== candle-mps-flash-attention benchmark ===\n");

    // Test with Swin-like dimensions (first stage)
    // 1024x1024 input -> 256x256 after patch_embed -> window_size=12
    // num_windows = ceil(256/12)^2 = 22^2 = 484
    let batch = 484;
    let num_heads = 6;
    let seq_len = 144; // 12x12 window
    let head_dim = 32;

    println!("Config: batch={}, heads={}, seq={}, head_dim={}", batch, num_heads, seq_len, head_dim);
    println!("Total elements: {}", batch * num_heads * seq_len * head_dim);

    let q = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), &device)?;
    let k = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), &device)?;

    // Warmup
    println!("\nWarmup...");
    for _ in 0..3 {
        let _ = flash_attention(&q, &k, &v, false)?;
    }
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }

    // Benchmark
    println!("\nBenchmarking (10 iterations)...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = flash_attention(&q, &k, &v, false)?;
    }
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    let elapsed = start.elapsed();
    println!("Flash attention: {:.2}ms/iter", elapsed.as_secs_f64() * 100.0);

    // Check output correctness
    let output = flash_attention(&q, &k, &v, false)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }

    let out_flat = output.flatten_all()?.to_vec1::<f32>()?;
    let min_val = out_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = out_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = out_flat.iter().sum();
    println!("\nOutput stats: min={:.4}, max={:.4}, sum={:.4}", min_val, max_val, sum);

    // Test different head dimensions
    println!("\n=== Head dimension comparison ===");
    for hd in [32, 64, 128] {
        let q = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, hd), &device)?;
        let k = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, hd), &device)?;
        let v = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, hd), &device)?;

        // Warmup
        let _ = flash_attention(&q, &k, &v, false)?;
        if let Device::Metal(m) = &device { m.wait_until_completed()?; }

        let start = Instant::now();
        for _ in 0..10 {
            let _ = flash_attention(&q, &k, &v, false)?;
        }
        if let Device::Metal(m) = &device { m.wait_until_completed()?; }
        let elapsed = start.elapsed();
        println!("head_dim={}: {:.2}ms/iter", hd, elapsed.as_secs_f64() * 100.0);
    }

    Ok(())
}
