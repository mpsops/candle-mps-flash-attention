# candle-mps-flash-attention

Flash Attention for [candle](https://github.com/huggingface/candle) on Apple Silicon (M1/M2/M3/M4).

**O(N) memory** instead of O(N²), enabling long sequences on unified memory.

## Features

- Flash attention forward pass (FP16/FP32)
- Additive attention bias (relative position encodings, ALiBi)
- Causal masking
- Sliding window attention
- Zero-copy integration via Metal command encoder

## Installation

```toml
[dependencies]
candle-mps-flash-attention = "0.1"
```

Requires `libMFABridge.dylib` from [mps-flash-attention](https://github.com/mpsops/mps-flash-attention):

```bash
# Build the Swift bridge
git clone --recursive https://github.com/mpsops/mps-flash-attention
cd mps-flash-attention/swift-bridge
swift build -c release

# Set path
export MFA_BRIDGE_PATH=$PWD/.build/release/libMFABridge.dylib
```

## Usage

```rust
use candle_mps_flash_attention::flash_attention;
use candle_core::{Device, Tensor};

let device = Device::new_metal(0)?;

// (B, H, N, D) format
let q = Tensor::randn(0., 1., (2, 8, 4096, 64), &device)?;
let k = Tensor::randn(0., 1., (2, 8, 4096, 64), &device)?;
let v = Tensor::randn(0., 1., (2, 8, 4096, 64), &device)?;

let out = flash_attention(&q, &k, &v, false)?;
```

### With Attention Bias

```rust
use candle_mps_flash_attention::flash_attention_with_bias;

// Position bias (must be pre-scaled by sqrt(head_dim))
let bias = Tensor::randn(0., 1., (1, 8, 64, 64), &device)?;
let scale = (64.0_f32).sqrt();
let scaled_bias = (&bias * scale)?;

let out = flash_attention_with_bias(&q, &k, &v, &scaled_bias, false)?;
```

### With Repeating Bias (Window Attention)

```rust
use candle_mps_flash_attention::flash_attention_with_repeating_bias;

// For Swin-style window attention where bias repeats every n_windows batches
let n_windows = 16;
let bias = Tensor::randn(0., 1., (n_windows, 8, 49, 49), &device)?;
let scaled_bias = (&bias * scale)?;

let out = flash_attention_with_repeating_bias(&q, &k, &v, &scaled_bias, n_windows, false)?;
```

## Requirements

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.70+

## Credits

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner
- [mps-flash-attention](https://github.com/mpsops/mps-flash-attention) Python/PyTorch bindings

## License

MIT
