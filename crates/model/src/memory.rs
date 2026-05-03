use std::collections::VecDeque;

/// Compressed long-context memory for extended histories.
#[derive(Clone, Debug)]
pub struct TemporalMemory {
    capacity: usize,
    slots: VecDeque<Vec<f32>>,
}

impl TemporalMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, summary: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        if self.slots.len() == self.capacity {
            self.slots.pop_front();
        }
        self.slots.push_back(summary);
    }

    pub fn summaries(&self) -> impl Iterator<Item = &[f32]> {
        self.slots.iter().map(Vec::as_slice)
    }
}
