// Modified work: small layers used by the ported SAM 2 modules (Apache-2.0, see NOTICE).
//! Linear, layer norm and a few tensor helpers on candle.

use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Tensor};
use candle_nn::{Linear, Module};

use crate::weights::Weights;
use crate::{Error, Result};

pub fn linear(w: &Weights, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    Ok(Linear::new(w.get(&format!("{prefix}.weight"), &[output, input])?, Some(w.get(&format!("{prefix}.bias"), &[output])?)))
}

/// Layer norm over the last dimension using candle's fused kernel (much faster than the
/// composed `candle_nn::LayerNorm` on large activations).
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(&x.contiguous()?, &self.weight, &self.bias, self.eps)?)
    }
}

pub fn layer_norm(w: &Weights, prefix: &str, dim: usize, eps: f64) -> Result<LayerNorm> {
    Ok(LayerNorm { weight: w.get(&format!("{prefix}.weight"), &[dim])?, bias: w.get(&format!("{prefix}.bias"), &[dim])?, eps: eps as f32 })
}

/// Two-layer MLP with exact (erf) GELU, as `Sam2FeedForward` with `num_layers = 2`.
pub struct Mlp {
    pub proj_in: Linear,
    pub proj_out: Linear,
}

impl Mlp {
    pub fn load(w: &Weights, prefix: &str, dim: usize, hidden: usize) -> Result<Mlp> {
        Ok(Mlp { proj_in: linear(w, &format!("{prefix}.proj_in"), dim, hidden)?, proj_out: linear(w, &format!("{prefix}.proj_out"), hidden, dim)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        apply(&self.proj_out, &gelu(&apply(&self.proj_in, x)?)?)
    }
}

/// Apply a linear layer to `[.., in]` through a flat 2-D matmul. Candle's `Linear::forward`
/// broadcasts the weight over every leading dimension (copying it per window), which is slow
/// on the 4-D activations of the backbone.
pub fn apply(l: &Linear, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (&last, lead) = dims.split_last().ok_or_else(|| Error::Compute("linear on a scalar".into()))?;
    let rows: usize = lead.iter().product();
    let y = l.forward(&x.reshape((rows, last))?)?;
    let mut out: Vec<usize> = lead.to_vec();
    out.push(y.dim(1)?);
    Ok(y.reshape(out)?)
}

/// Exact (erf) GELU on all cores. Candle's own `gelu_erf` runs on one thread, which made it a third of
/// the time of the Hiera blocks; this gives the same values (it calls the same `erf` per element).
struct GeluErf;

impl CustomOp1 for GeluErf {
    fn name(&self) -> &'static str {
        "gelu_erf_parallel"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        use rayon::prelude::*;
        let CpuStorage::F32(data) = storage else { candle_core::bail!("gelu_erf_parallel needs f32") };
        let Some((start, end)) = layout.contiguous_offsets() else { candle_core::bail!("gelu_erf_parallel needs a contiguous input") };
        let src = data.get(start..end).ok_or_else(|| candle_core::Error::Msg("gelu_erf_parallel: layout outside the storage".into()))?;
        let mut out = vec![0f32; src.len()];
        const CHUNK: usize = 16 * 1024;
        out.par_chunks_mut(CHUNK).zip(src.par_chunks(CHUNK)).for_each(|(o, i)| {
            for (a, v) in o.iter_mut().zip(i) {
                *a = (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.0) * 0.5 * v;
            }
        });
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

/// Exact GELU of `x`, in parallel for f32 tensors.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F32 || !x.device().is_cpu() {
        return Ok(x.gelu_erf()?);
    }
    Ok(x.contiguous()?.apply_op1_no_bwd(&GeluErf)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn the_parallel_gelu_equals_candles_bit_for_bit() {
        let v: Vec<f32> = (0..101_000).map(|i| ((i as f32) * 0.00037).sin() * 6.0 - 0.5).collect();
        let x = Tensor::from_vec(v, (101, 1000), &Device::Cpu).unwrap();
        let (a, b) = (gelu(&x).unwrap(), x.gelu_erf().unwrap());
        assert_eq!(a.flatten_all().unwrap().to_vec1::<f32>().unwrap(), b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        // a transposed (non-contiguous) input works, and an empty one does too
        let t = x.t().unwrap();
        assert_eq!(gelu(&t).unwrap().dims(), t.dims());
        let e = Tensor::zeros((0, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_eq!(gelu(&e).unwrap().elem_count(), 0);
    }
}
