use serde::{Deserialize, Serialize};

/// Missing-value policy used before patching or model inference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingPolicy {
    /// Preserve missing values as `NaN`; downstream masks remain informative.
    #[default]
    Preserve,
    /// Replace missing values with zero.
    ZeroFill,
    /// Carry the last observed value forward.
    ForwardFill,
    /// Linearly interpolate interior gaps and forward/backward fill boundary gaps.
    LinearInterpolate,
}

/// Result of applying a missing-value policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MissingValueTreatment {
    pub values: Vec<f32>,
    pub observed: Vec<bool>,
}

impl MissingPolicy {
    pub fn apply(self, values: &[f32]) -> MissingValueTreatment {
        let observed: Vec<bool> = values.iter().map(|value| value.is_finite()).collect();
        let mut treated = values.to_vec();
        match self {
            Self::Preserve => {}
            Self::ZeroFill => {
                for value in &mut treated {
                    if !value.is_finite() {
                        *value = 0.0;
                    }
                }
            }
            Self::ForwardFill => forward_fill(&mut treated),
            Self::LinearInterpolate => interpolate(&mut treated),
        }
        MissingValueTreatment {
            values: treated,
            observed,
        }
    }
}

fn forward_fill(values: &mut [f32]) {
    let mut last = None;
    for value in values.iter_mut() {
        if value.is_finite() {
            last = Some(*value);
        } else if let Some(previous) = last {
            *value = previous;
        }
    }
    backfill_leading(values);
}

fn interpolate(values: &mut [f32]) {
    let mut index = 0;
    while index < values.len() {
        if values[index].is_finite() {
            index += 1;
            continue;
        }
        let start = index;
        while index < values.len() && !values[index].is_finite() {
            index += 1;
        }
        let end = index;
        let left = start
            .checked_sub(1)
            .and_then(|left| values.get(left))
            .copied();
        let right = values.get(end).copied();
        match (left, right) {
            (Some(left), Some(right)) if left.is_finite() && right.is_finite() => {
                let span = (end - start + 1) as f32;
                for (offset, value) in values[start..end].iter_mut().enumerate() {
                    let weight = (offset + 1) as f32 / span;
                    *value = left + (right - left) * weight;
                }
            }
            (Some(left), _) if left.is_finite() => {
                for value in &mut values[start..end] {
                    *value = left;
                }
            }
            (_, Some(right)) if right.is_finite() => {
                for value in &mut values[start..end] {
                    *value = right;
                }
            }
            _ => {}
        }
    }
    backfill_leading(values);
}

fn backfill_leading(values: &mut [f32]) {
    if let Some(first_observed) = values.iter().copied().find(|value| value.is_finite()) {
        for value in values.iter_mut().take_while(|value| !value.is_finite()) {
            *value = first_observed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_interior_gap_and_preserves_mask() {
        let result = MissingPolicy::LinearInterpolate.apply(&[1.0, f32::NAN, 3.0]);
        assert_eq!(result.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.observed, vec![true, false, true]);
    }
}
