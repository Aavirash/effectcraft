// Modified work: small layers used by the ported SAM 2 modules (Apache-2.0, see NOTICE).
//! Linear, layer norm and a few tensor helpers on candle.

use candle_core::Tensor;
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
        apply(&self.proj_out, &apply(&self.proj_in, x)?.gelu_erf()?)
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
