// Modified work: small layers used by the ported SAM 2 modules (Apache-2.0, see NOTICE).
//! Linear, layer norm and a few tensor helpers on candle.

use candle_core::Tensor;
use candle_nn::{LayerNorm, Linear, Module};

use crate::Result;
use crate::weights::Weights;

pub fn linear(w: &Weights, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    Ok(Linear::new(w.get(&format!("{prefix}.weight"), &[output, input])?, Some(w.get(&format!("{prefix}.bias"), &[output])?)))
}

pub fn layer_norm(w: &Weights, prefix: &str, dim: usize, eps: f64) -> Result<LayerNorm> {
    Ok(LayerNorm::new(w.get(&format!("{prefix}.weight"), &[dim])?, w.get(&format!("{prefix}.bias"), &[dim])?, eps))
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
        Ok(self.proj_out.forward(&self.proj_in.forward(x)?.gelu_erf()?)?)
    }
}
