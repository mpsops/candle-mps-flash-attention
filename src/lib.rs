//! MPS Flash Attention for candle
//!
//! This crate provides flash attention on Apple Silicon via the metal-flash-attention library.
//! It dynamically loads libMFABridge.dylib which contains the Swift implementation.

use candle_core::{DType, Device, Result, Tensor};
use once_cell::sync::OnceCell;
use std::ffi::c_void;
use std::ptr;

// FFI function types matching MFABridge.swift
type MfaInitFn = unsafe extern "C" fn() -> bool;
type MfaCreateKernelV5Fn = unsafe extern "C" fn(
    seq_len_q: i32,
    seq_len_kv: i32,
    head_dim: i32,
    low_precision: bool,
    low_precision_outputs: bool,
    causal: bool,
    has_mask: bool,
    use_bf16: bool,
    window_size: u32,
    quantized_kv: u16, // 0=none, 1=FP8_E4M3, etc.
    bf16_backward: bool,
) -> *mut c_void;
type MfaCreateKernelV6Fn = unsafe extern "C" fn(
    seq_len_q: i32,
    seq_len_kv: i32,
    head_dim: i32,
    low_precision: bool,
    low_precision_outputs: bool,
    causal: bool,
    has_mask: bool,
    use_bf16: bool,
    window_size: u32,
    quantized_kv: u16,
    bf16_backward: bool,
    has_attn_bias: bool,
    bias_batch_stride: u32,
    bias_head_stride: u32,
) -> *mut c_void;
type MfaCreateKernelV7Fn = unsafe extern "C" fn(
    seq_len_q: i32,
    seq_len_kv: i32,
    head_dim: i32,
    low_precision: bool,
    low_precision_outputs: bool,
    causal: bool,
    has_mask: bool,
    use_bf16: bool,
    window_size: u32,
    quantized_kv: u16,
    bf16_backward: bool,
    has_attn_bias: bool,
    bias_batch_stride: u32,
    bias_head_stride: u32,
    bias_repeat_count: u32,
) -> *mut c_void;
type MfaForwardEncodeFn = unsafe extern "C" fn(
    kernel_handle: *mut c_void,
    encoder_ptr: *mut c_void,  // MTLComputeCommandEncoder
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    l_ptr: *mut c_void,
    mask_ptr: *mut c_void,
    q_offset: i64,
    k_offset: i64,
    v_offset: i64,
    o_offset: i64,
    l_offset: i64,
    mask_offset: i64,
    batch_size: i32,
    num_heads: i32,
) -> bool;
type MfaForwardEncodeBiasFn = unsafe extern "C" fn(
    kernel_handle: *mut c_void,
    encoder_ptr: *mut c_void,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    l_ptr: *mut c_void,
    mask_ptr: *mut c_void,
    attn_bias_raw_ptr: *mut c_void,
    q_offset: i64,
    k_offset: i64,
    v_offset: i64,
    o_offset: i64,
    l_offset: i64,
    mask_offset: i64,
    attn_bias_offset: i64,
    batch_size: i32,
    num_heads: i32,
) -> bool;
type MfaReleaseKernelFn = unsafe extern "C" fn(kernel_handle: *mut c_void);

// Global state
static MFA_LIB: OnceCell<MfaLibrary> = OnceCell::new();

#[allow(dead_code)]
struct MfaLibrary {
    _handle: *mut c_void,
    init: MfaInitFn,
    create_kernel_v5: MfaCreateKernelV5Fn,
    create_kernel_v6: MfaCreateKernelV6Fn,
    create_kernel_v7: MfaCreateKernelV7Fn,
    forward_encode: MfaForwardEncodeFn,
    forward_encode_bias: MfaForwardEncodeBiasFn,
    release_kernel: MfaReleaseKernelFn,
}

// Safety: The library handle and function pointers are thread-safe to share
unsafe impl Send for MfaLibrary {}
unsafe impl Sync for MfaLibrary {}

impl MfaLibrary {
    fn load() -> std::result::Result<Self, String> {
        // Try to find libMFABridge.dylib
        //
        // Users must set MFA_BRIDGE_PATH to the path of libMFABridge.dylib.
        // Build it from: https://github.com/mpsops/mps-flash-attention
        //   cd swift-bridge && swift build -c release
        //   export MFA_BRIDGE_PATH=$PWD/.build/release/libMFABridge.dylib

        let path = std::env::var("MFA_BRIDGE_PATH").map_err(|_| {
            "MFA_BRIDGE_PATH environment variable not set. \
             Build libMFABridge.dylib from https://github.com/mpsops/mps-flash-attention: \
             cd swift-bridge && swift build -c release && \
             export MFA_BRIDGE_PATH=$PWD/.build/release/libMFABridge.dylib".to_string()
        })?;

        let c_path = std::ffi::CString::new(path.as_str()).unwrap();
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW) };

        if handle.is_null() {
            let err = unsafe { std::ffi::CStr::from_ptr(libc::dlerror()) };
            return Err(format!("Failed to load libMFABridge.dylib from '{}': {:?}", path, err));
        }

        // Load function symbols
        let init = unsafe {
            let sym = libc::dlsym(handle, b"mfa_init\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_init".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaInitFn>(sym)
        };

        let create_kernel_v5 = unsafe {
            let sym = libc::dlsym(handle, b"mfa_create_kernel_v5\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_create_kernel_v5".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaCreateKernelV5Fn>(sym)
        };

        let create_kernel_v6 = unsafe {
            let sym = libc::dlsym(handle, b"mfa_create_kernel_v6\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_create_kernel_v6".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaCreateKernelV6Fn>(sym)
        };

        let create_kernel_v7 = unsafe {
            let sym = libc::dlsym(handle, b"mfa_create_kernel_v7\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_create_kernel_v7".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaCreateKernelV7Fn>(sym)
        };

        let forward_encode = unsafe {
            let sym = libc::dlsym(handle, b"mfa_forward_encode\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_forward_encode".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaForwardEncodeFn>(sym)
        };

        let forward_encode_bias = unsafe {
            let sym = libc::dlsym(handle, b"mfa_forward_encode_bias\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_forward_encode_bias".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaForwardEncodeBiasFn>(sym)
        };

        let release_kernel = unsafe {
            let sym = libc::dlsym(handle, b"mfa_release_kernel\0".as_ptr() as *const i8);
            if sym.is_null() {
                return Err("Failed to find mfa_release_kernel".to_string());
            }
            std::mem::transmute::<*mut c_void, MfaReleaseKernelFn>(sym)
        };

        // Initialize
        if !unsafe { init() } {
            return Err("mfa_init() failed".to_string());
        }

        Ok(Self {
            _handle: handle,
            init,
            create_kernel_v5,
            create_kernel_v6,
            create_kernel_v7,
            forward_encode,
            forward_encode_bias,
            release_kernel,
        })
    }
}

fn get_mfa() -> Result<&'static MfaLibrary> {
    MFA_LIB.get_or_try_init(|| {
        MfaLibrary::load().map_err(|e| candle_core::Error::Msg(e))
    })
}

/// Kernel cache for reusing compiled kernels
struct KernelHandle {
    ptr: *mut c_void,
}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        if let Ok(mfa) = get_mfa() {
            unsafe { (mfa.release_kernel)(self.ptr) };
        }
    }
}

// Safety: The kernel handle is thread-safe (contains compiled Metal pipelines)
unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}

/// Flash attention forward pass
///
/// # Arguments
/// * `query` - Query tensor [batch, num_heads, seq_len_q, head_dim]
/// * `key` - Key tensor [batch, num_heads, seq_len_kv, head_dim]
/// * `value` - Value tensor [batch, num_heads, seq_len_kv, head_dim]
/// * `is_causal` - Whether to apply causal masking
///
/// # Returns
/// Output tensor [batch, num_heads, seq_len_q, head_dim]
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    is_causal: bool,
) -> Result<Tensor> {
    flash_attention_with_mask(query, key, value, is_causal, None, 0)
}

/// Flash attention forward pass with optional mask and sliding window
pub fn flash_attention_with_mask(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    is_causal: bool,
    mask: Option<&Tensor>,
    window_size: u32,
) -> Result<Tensor> {
    let mfa = get_mfa()?;

    // Validate inputs
    let device = query.device();
    let Device::Metal(metal_device) = device else {
        return Err(candle_core::Error::Msg("flash_attention requires Metal device".to_string()));
    };

    let dtype = query.dtype();
    let low_precision = dtype == DType::F16;
    let use_bf16 = dtype == DType::BF16;

    if dtype != DType::F32 && dtype != DType::F16 && dtype != DType::BF16 {
        return Err(candle_core::Error::Msg(format!(
            "flash_attention requires F32, F16, or BF16, got {:?}",
            dtype
        )));
    }

    // Get dimensions [batch, num_heads, seq_len, head_dim]
    let (batch, num_heads, seq_len_q, head_dim) = query.dims4()?;
    let (_, _, seq_len_kv, _) = key.dims4()?;

    // Create kernel
    let kernel_handle = unsafe {
        (mfa.create_kernel_v5)(
            seq_len_q as i32,
            seq_len_kv as i32,
            head_dim as i32,
            low_precision,
            low_precision, // low_precision_outputs
            is_causal,
            mask.is_some(),
            use_bf16,
            window_size,
            0, // no quantization
            false, // bf16_backward
        )
    };

    if kernel_handle.is_null() {
        return Err(candle_core::Error::Msg("Failed to create MFA kernel".to_string()));
    }

    let _kernel = KernelHandle { ptr: kernel_handle };

    // Ensure tensors are contiguous
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;

    // Create output tensor
    let output = Tensor::zeros(query.dims(), dtype, device)?;

    // Create logsumexp tensor (needed by MFA, used for backward pass)
    let logsumexp = Tensor::zeros((batch, num_heads, seq_len_q), DType::F32, device)?;

    // Get Metal buffers - convert to raw pointers for FFI
    fn buffer_to_ptr(storage: &candle_core::Storage) -> Result<*mut c_void> {
        match storage {
            candle_core::Storage::Metal(s) => {
                // Get the raw MTLBuffer pointer for FFI
                Ok(s.buffer().as_raw_ptr() as *mut c_void)
            }
            _ => Err(candle_core::Error::Msg("Expected Metal storage".to_string())),
        }
    }

    let (q_storage, q_layout) = query.storage_and_layout();
    let (k_storage, k_layout) = key.storage_and_layout();
    let (v_storage, v_layout) = value.storage_and_layout();
    let (o_storage, _o_layout) = output.storage_and_layout();
    let (l_storage, _l_layout) = logsumexp.storage_and_layout();

    let q_buf = buffer_to_ptr(&q_storage)?;
    let k_buf = buffer_to_ptr(&k_storage)?;
    let v_buf = buffer_to_ptr(&v_storage)?;
    let o_buf = buffer_to_ptr(&o_storage)?;
    let l_buf = buffer_to_ptr(&l_storage)?;

    let elem_size = dtype.size_in_bytes();
    let q_offset = (q_layout.start_offset() * elem_size) as i64;
    let k_offset = (k_layout.start_offset() * elem_size) as i64;
    let v_offset = (v_layout.start_offset() * elem_size) as i64;

    // Mask handling
    let mask_contiguous;
    let mask_storage_ref;
    let (mask_buf, mask_offset) = if let Some(m) = mask {
        mask_contiguous = m.contiguous()?;
        let (m_storage, m_layout) = mask_contiguous.storage_and_layout();
        mask_storage_ref = Some(m_storage);
        let buf = buffer_to_ptr(mask_storage_ref.as_ref().unwrap())?;
        (buf, (m_layout.start_offset() * mask_contiguous.dtype().size_in_bytes()) as i64)
    } else {
        (ptr::null_mut(), 0i64)
    };
    let _ = mask_storage_ref; // keep alive until after FFI call

    // Get candle's command encoder - this ensures MFA work is on the same command buffer
    let encoder = metal_device.command_encoder()?;
    let encoder_ptr = encoder.as_raw_ptr() as *mut c_void;

    // Call MFA forward_encode (uses the provided encoder)
    let success = unsafe {
        (mfa.forward_encode)(
            kernel_handle,
            encoder_ptr,
            q_buf,
            k_buf,
            v_buf,
            o_buf,
            l_buf,
            mask_buf,
            q_offset,
            k_offset,
            v_offset,
            0, // o_offset
            0, // l_offset
            mask_offset,
            batch as i32,
            num_heads as i32,
        )
    };

    // Drop the encoder to end encoding (important!)
    drop(encoder);

    // Drop storage refs
    drop(q_storage);
    drop(k_storage);
    drop(v_storage);
    drop(o_storage);
    drop(l_storage);

    if !success {
        return Err(candle_core::Error::Msg("MFA forward_encode returned false".to_string()));
    }

    // Wait for completion - this commits and waits for the command buffer
    metal_device.wait_until_completed()?;

    Ok(output)
}

/// Flash attention forward pass with additive attention bias
///
/// # Arguments
/// * `query` - Query tensor [batch, num_heads, seq_len_q, head_dim]
/// * `key` - Key tensor [batch, num_heads, seq_len_kv, head_dim]
/// * `value` - Value tensor [batch, num_heads, seq_len_kv, head_dim]
/// * `attn_bias` - Attention bias [batch, num_heads, seq_len_q, seq_len_kv] or broadcastable
/// * `is_causal` - Whether to apply causal masking
///
/// # Returns
/// Output tensor [batch, num_heads, seq_len_q, head_dim]
pub fn flash_attention_with_bias(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_bias: &Tensor,
    is_causal: bool,
) -> Result<Tensor> {
    let mfa = get_mfa()?;

    // Validate inputs
    let device = query.device();
    let Device::Metal(metal_device) = device else {
        return Err(candle_core::Error::Msg("flash_attention requires Metal device".to_string()));
    };

    let dtype = query.dtype();
    let low_precision = dtype == DType::F16;
    let use_bf16 = dtype == DType::BF16;

    if dtype != DType::F32 && dtype != DType::F16 && dtype != DType::BF16 {
        return Err(candle_core::Error::Msg(format!(
            "flash_attention requires F32, F16, or BF16, got {:?}",
            dtype
        )));
    }

    // Get dimensions [batch, num_heads, seq_len, head_dim]
    let (batch, num_heads, seq_len_q, head_dim) = query.dims4()?;
    let (_, _, seq_len_kv, _) = key.dims4()?;

    // Determine bias strides for broadcasting
    let bias_dims = attn_bias.dims();
    let (bias_batch_stride, bias_head_stride) = if bias_dims.len() == 4 {
        let bias_batch = bias_dims[0];
        let bias_heads = bias_dims[1];
        // If bias batch == 1, broadcast across batch (stride = 0)
        // If bias heads == 1, broadcast across heads (stride = 0)
        let batch_stride = if bias_batch == 1 { 0 } else { (bias_heads * seq_len_q * seq_len_kv) as u32 };
        let head_stride = if bias_heads == 1 { 0 } else { (seq_len_q * seq_len_kv) as u32 };
        (batch_stride, head_stride)
    } else if bias_dims.len() == 3 {
        // [num_heads, seq_q, seq_k] - broadcast across batch
        let bias_heads = bias_dims[0];
        let head_stride = if bias_heads == 1 { 0 } else { (seq_len_q * seq_len_kv) as u32 };
        (0, head_stride)
    } else {
        return Err(candle_core::Error::Msg(format!(
            "attn_bias must be 3D or 4D, got {:?}",
            bias_dims
        )));
    };

    // Create kernel with bias support
    let kernel_handle = unsafe {
        (mfa.create_kernel_v6)(
            seq_len_q as i32,
            seq_len_kv as i32,
            head_dim as i32,
            low_precision,
            low_precision, // low_precision_outputs
            is_causal,
            false, // has_mask (boolean mask)
            use_bf16,
            0, // window_size
            0, // no quantization
            false, // bf16_backward
            true, // has_attn_bias
            bias_batch_stride,
            bias_head_stride,
        )
    };

    if kernel_handle.is_null() {
        return Err(candle_core::Error::Msg("Failed to create MFA kernel with bias".to_string()));
    }

    let _kernel = KernelHandle { ptr: kernel_handle };

    // Ensure tensors are contiguous
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let attn_bias = attn_bias.contiguous()?;

    // Create output tensor
    let output = Tensor::zeros(query.dims(), dtype, device)?;

    // Create logsumexp tensor (needed by MFA, used for backward pass)
    let logsumexp = Tensor::zeros((batch, num_heads, seq_len_q), DType::F32, device)?;

    // Get Metal buffers
    fn buffer_to_ptr(storage: &candle_core::Storage) -> Result<*mut c_void> {
        match storage {
            candle_core::Storage::Metal(s) => {
                Ok(s.buffer().as_raw_ptr() as *mut c_void)
            }
            _ => Err(candle_core::Error::Msg("Expected Metal storage".to_string())),
        }
    }

    let (q_storage, q_layout) = query.storage_and_layout();
    let (k_storage, k_layout) = key.storage_and_layout();
    let (v_storage, v_layout) = value.storage_and_layout();
    let (o_storage, _o_layout) = output.storage_and_layout();
    let (l_storage, _l_layout) = logsumexp.storage_and_layout();
    let (bias_storage, bias_layout) = attn_bias.storage_and_layout();

    let q_buf = buffer_to_ptr(&q_storage)?;
    let k_buf = buffer_to_ptr(&k_storage)?;
    let v_buf = buffer_to_ptr(&v_storage)?;
    let o_buf = buffer_to_ptr(&o_storage)?;
    let l_buf = buffer_to_ptr(&l_storage)?;
    let bias_buf = buffer_to_ptr(&bias_storage)?;

    let elem_size = dtype.size_in_bytes();
    let q_offset = (q_layout.start_offset() * elem_size) as i64;
    let k_offset = (k_layout.start_offset() * elem_size) as i64;
    let v_offset = (v_layout.start_offset() * elem_size) as i64;
    let bias_offset = (bias_layout.start_offset() * elem_size) as i64;

    // Get candle's command encoder
    let encoder = metal_device.command_encoder()?;
    let encoder_ptr = encoder.as_raw_ptr() as *mut c_void;

    // Call MFA forward_encode_bias
    let success = unsafe {
        (mfa.forward_encode_bias)(
            kernel_handle,
            encoder_ptr,
            q_buf,
            k_buf,
            v_buf,
            o_buf,
            l_buf,
            ptr::null_mut(), // no boolean mask
            bias_buf,
            q_offset,
            k_offset,
            v_offset,
            0, // o_offset
            0, // l_offset
            0, // mask_offset
            bias_offset,
            batch as i32,
            num_heads as i32,
        )
    };

    // Drop the encoder to end encoding
    drop(encoder);

    // Drop storage refs
    drop(q_storage);
    drop(k_storage);
    drop(v_storage);
    drop(o_storage);
    drop(l_storage);
    drop(bias_storage);

    if !success {
        return Err(candle_core::Error::Msg("MFA forward_encode_bias returned false".to_string()));
    }

    // Wait for completion
    metal_device.wait_until_completed()?;

    Ok(output)
}

/// Flash attention with bias that repeats across batch dimension
///
/// For window attention where bias pattern repeats every `repeat_count` batches.
/// bias shape: [repeat_count, num_heads, seq_q, seq_k]
/// The encoder will use batch_idx % repeat_count to index into the bias.
pub fn flash_attention_with_repeating_bias(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_bias: &Tensor,
    repeat_count: usize,
    is_causal: bool,
) -> Result<Tensor> {
    let mfa = get_mfa()?;

    let device = query.device();
    let Device::Metal(metal_device) = device else {
        return Err(candle_core::Error::Msg("flash_attention requires Metal device".to_string()));
    };

    let dtype = query.dtype();
    let low_precision = dtype == DType::F16;
    let use_bf16 = dtype == DType::BF16;

    if dtype != DType::F32 && dtype != DType::F16 && dtype != DType::BF16 {
        return Err(candle_core::Error::Msg(format!(
            "flash_attention requires F32, F16, or BF16, got {:?}",
            dtype
        )));
    }

    let (batch, num_heads, seq_len_q, head_dim) = query.dims4()?;
    let (_, _, seq_len_kv, _) = key.dims4()?;

    // Bias shape should be [repeat_count, num_heads, seq_q, seq_k]
    let bias_dims = attn_bias.dims();
    let bias_head_stride = if bias_dims.len() >= 2 {
        (seq_len_q * seq_len_kv) as u32
    } else {
        0
    };

    // Create kernel with bias repeat support
    let kernel_handle = unsafe {
        (mfa.create_kernel_v7)(
            seq_len_q as i32,
            seq_len_kv as i32,
            head_dim as i32,
            low_precision,
            low_precision,
            is_causal,
            false, // has_mask
            use_bf16,
            0, // window_size
            0, // no quantization
            false, // bf16_backward
            true, // has_attn_bias
            0, // bias_batch_stride (not used when repeat_count > 0)
            bias_head_stride,
            repeat_count as u32, // bias_repeat_count
        )
    };

    if kernel_handle.is_null() {
        return Err(candle_core::Error::Msg("Failed to create MFA kernel with repeating bias".to_string()));
    }

    let _kernel = KernelHandle { ptr: kernel_handle };

    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let attn_bias = attn_bias.contiguous()?;

    let output = Tensor::zeros(query.dims(), dtype, device)?;
    let logsumexp = Tensor::zeros((batch, num_heads, seq_len_q), DType::F32, device)?;

    fn buffer_to_ptr(storage: &candle_core::Storage) -> Result<*mut c_void> {
        match storage {
            candle_core::Storage::Metal(s) => Ok(s.buffer().as_raw_ptr() as *mut c_void),
            _ => Err(candle_core::Error::Msg("Expected Metal storage".to_string())),
        }
    }

    let (q_storage, q_layout) = query.storage_and_layout();
    let (k_storage, k_layout) = key.storage_and_layout();
    let (v_storage, v_layout) = value.storage_and_layout();
    let (o_storage, _) = output.storage_and_layout();
    let (l_storage, _) = logsumexp.storage_and_layout();
    let (bias_storage, bias_layout) = attn_bias.storage_and_layout();

    let q_buf = buffer_to_ptr(&q_storage)?;
    let k_buf = buffer_to_ptr(&k_storage)?;
    let v_buf = buffer_to_ptr(&v_storage)?;
    let o_buf = buffer_to_ptr(&o_storage)?;
    let l_buf = buffer_to_ptr(&l_storage)?;
    let bias_buf = buffer_to_ptr(&bias_storage)?;

    let elem_size = dtype.size_in_bytes();
    let q_offset = (q_layout.start_offset() * elem_size) as i64;
    let k_offset = (k_layout.start_offset() * elem_size) as i64;
    let v_offset = (v_layout.start_offset() * elem_size) as i64;
    let bias_offset = (bias_layout.start_offset() * elem_size) as i64;

    let encoder = metal_device.command_encoder()?;
    let encoder_ptr = encoder.as_raw_ptr() as *mut c_void;

    let success = unsafe {
        (mfa.forward_encode_bias)(
            kernel_handle,
            encoder_ptr,
            q_buf, k_buf, v_buf, o_buf, l_buf,
            ptr::null_mut(), // no boolean mask
            bias_buf,
            q_offset, k_offset, v_offset,
            0, 0, 0, // o, l, mask offsets
            bias_offset,
            batch as i32,
            num_heads as i32,
        )
    };

    drop(encoder);
    drop(q_storage);
    drop(k_storage);
    drop(v_storage);
    drop(o_storage);
    drop(l_storage);
    drop(bias_storage);

    if !success {
        return Err(candle_core::Error::Msg("MFA forward_encode_bias returned false".to_string()));
    }

    metal_device.wait_until_completed()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_basic() -> Result<()> {
        let device = Device::new_metal(0)?;

        let batch = 1;
        let num_heads = 8;
        let seq_len = 64;
        let head_dim = 64;

        let q = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)?;

        let output = flash_attention(&q, &k, &v, false)?;

        assert_eq!(output.dims(), &[batch, num_heads, seq_len, head_dim]);

        Ok(())
    }
}
