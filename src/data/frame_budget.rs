//! Frame-time feedback controls work, with a dead band and slow recovery to avoid
//! quality oscillation. The byte cache budget remains an independent hard limit.
#[derive(Debug)]
pub struct FrameBudget {
    budget: u64,
    maximum: u64,
    sum_ms: f64,
    samples: usize,
    elapsed_s: f64,
}
impl FrameBudget {
    pub fn new(maximum: u64) -> Self { Self { budget: maximum, maximum, sum_ms: 0.0, samples: 0, elapsed_s: 0.0 } }
    pub fn current(&self) -> u64 { self.budget }
    pub fn observe(&mut self, frame_ms: f64, foreground: bool, maximum: u64) {
        self.maximum = maximum; self.budget = self.budget.min(maximum);
        if !foreground || !frame_ms.is_finite() || frame_ms < 1.0 { return; }
        let sample = frame_ms.min(100.0);
        self.sum_ms += sample; self.samples += 1; self.elapsed_s += sample * 0.001;
        if self.elapsed_s < 0.5 || self.samples < 5 { return; }
        let average = self.sum_ms / self.samples as f64;
        if average > 20.0 {
            self.budget = ((self.budget as f64 * 0.82) as u64).max(350_000.min(maximum));
        } else if average < 17.5 && self.budget < maximum {
            self.budget = ((self.budget as f64 * 1.15) as u64).min(maximum);
        }
        self.sum_ms = 0.0; self.samples = 0; self.elapsed_s = 0.0;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sustained_slow_frames_reduce_work_and_fast_frames_recover() {
        let mut b = FrameBudget::new(3_000_000);
        for _ in 0..120 { b.observe(33.3, true, 3_000_000); }
        assert!(b.current() < 1_000_000 && b.current() >= 350_000);
        for _ in 0..2400 { b.observe(8.33, true, 3_000_000); }
        assert_eq!(b.current(), 3_000_000);
    }
    #[test]
    fn background_throttling_and_single_stalls_do_not_lower_quality() {
        let mut b = FrameBudget::new(3_000_000);
        for _ in 0..300 { b.observe(40.0, false, 3_000_000); }
        b.observe(200.0, true, 3_000_000);
        for _ in 0..120 { b.observe(8.33, true, 3_000_000); }
        assert_eq!(b.current(), 3_000_000);
    }
}
