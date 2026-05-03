/// Gated fusion between continuous value, discrete regime, and covariate streams.
#[derive(Clone, Debug)]
pub struct GatedFusion {
    value_weight: f32,
    regime_weight: f32,
    covariate_weight: f32,
}

impl Default for GatedFusion {
    fn default() -> Self {
        Self {
            value_weight: 0.6,
            regime_weight: 0.3,
            covariate_weight: 0.1,
        }
    }
}

impl GatedFusion {
    pub fn fuse(&self, value: &[f32], regime: &[f32], covariate: Option<&[f32]>) -> Vec<f32> {
        let len = value
            .len()
            .max(regime.len())
            .max(covariate.map_or(0, <[f32]>::len));
        let mut fused = Vec::with_capacity(len);
        for index in 0..len {
            let value_part = value.get(index).copied().unwrap_or_default() * self.value_weight;
            let regime_part = regime.get(index).copied().unwrap_or_default() * self.regime_weight;
            let covariate_part = covariate
                .and_then(|items| items.get(index))
                .copied()
                .unwrap_or_default()
                * self.covariate_weight;
            fused.push(value_part + regime_part + covariate_part);
        }
        fused
    }
}
