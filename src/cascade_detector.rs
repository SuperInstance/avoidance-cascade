//! Cascade detection for monitoring avoidance ratios in agent populations.

/// Monitors a population's avoidance ratio and detects cascade events.
///
/// A **cascade event** occurs when the fraction of agents choosing "avoid"
/// exceeds a configurable threshold (default 0.95). The detector tracks
/// consecutive rounds above threshold to confirm a sustained cascade.
#[derive(Debug, Clone)]
pub struct CascadeDetector {
    /// The avoidance ratio threshold that triggers cascade detection.
    threshold: f64,
    /// Number of consecutive rounds above threshold required to confirm cascade.
    confirmation_rounds: u32,
    /// Current consecutive rounds above threshold.
    consecutive_above: u32,
    /// Whether a cascade is currently active.
    cascade_active: bool,
    /// Total cascade events detected.
    total_cascades: u64,
    /// Total recovery events (cascade ended).
    total_recoveries: u64,
    /// History of avoidance ratios.
    ratio_history: Vec<f64>,
}

impl CascadeDetector {
    /// Creates a new detector with the given threshold.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not in `(0.0, 1.0]`.
    pub fn new(threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold <= 1.0,
            "threshold must be in (0.0, 1.0]"
        );
        Self {
            threshold,
            confirmation_rounds: 1,
            consecutive_above: 0,
            cascade_active: false,
            total_cascades: 0,
            total_recoveries: 0,
            ratio_history: Vec::new(),
        }
    }

    /// Creates a detector with default threshold of 0.95.
    pub fn with_default_threshold() -> Self {
        Self::new(0.95)
    }

    /// Sets the number of consecutive rounds above threshold to confirm cascade.
    pub fn with_confirmation_rounds(mut self, rounds: u32) -> Self {
        self.confirmation_rounds = rounds.max(1);
        self
    }

    /// Returns the cascade threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns whether a cascade is currently active.
    pub fn is_cascade_active(&self) -> bool {
        self.cascade_active
    }

    /// Returns the total number of cascade events detected.
    pub fn total_cascades(&self) -> u64 {
        self.total_cascades
    }

    /// Returns the total number of recovery events.
    pub fn total_recoveries(&self) -> u64 {
        self.total_recoveries
    }

    /// Returns the history of observed avoidance ratios.
    pub fn ratio_history(&self) -> &[f64] {
        &self.ratio_history
    }

    /// Computes the avoidance ratio from agent counts.
    ///
    /// Returns `avoid_count / total_agents`, or 0.0 if `total_agents` is 0.
    pub fn avoidance_ratio(total_agents: usize, avoid_count: usize) -> f64 {
        if total_agents == 0 {
            0.0
        } else {
            avoid_count as f64 / total_agents as f64
        }
    }

    /// Checks the current round's avoidance ratio.
    ///
    /// Returns `true` if a cascade alert is raised (transition to active).
    ///
    /// This method updates internal state and may trigger cascade/recovery events.
    pub fn check(&mut self, total_agents: usize, avoid_count: usize) -> bool {
        let ratio = Self::avoidance_ratio(total_agents, avoid_count);
        self.ratio_history.push(ratio);

        if ratio >= self.threshold {
            self.consecutive_above += 1;
        } else {
            self.consecutive_above = 0;
        }

        let confirmed = self.consecutive_above >= self.confirmation_rounds;

        if confirmed && !self.cascade_active {
            self.cascade_active = true;
            self.total_cascades += 1;
            return true; // alert raised
        }

        if !confirmed && self.cascade_active {
            self.cascade_active = false;
            self.total_recoveries += 1;
        }

        false
    }

    /// Resets all state.
    pub fn reset(&mut self) {
        self.consecutive_above = 0;
        self.cascade_active = false;
        self.total_cascades = 0;
        self.total_recoveries = 0;
        self.ratio_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_detector_default_state() {
        let d = CascadeDetector::new(0.9);
        assert!(!d.is_cascade_active());
        assert_eq!(d.total_cascades(), 0);
        assert_eq!(d.total_recoveries(), 0);
        assert!((d.threshold() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn zero_threshold_panics() {
        CascadeDetector::new(0.0);
    }

    #[test]
    #[should_panic]
    fn over_one_threshold_panics() {
        CascadeDetector::new(1.1);
    }

    #[test]
    fn detects_cascade_immediately() {
        let mut d = CascadeDetector::new(0.95);
        let alert = d.check(100, 96); // 0.96 > 0.95
        assert!(alert);
        assert!(d.is_cascade_active());
        assert_eq!(d.total_cascades(), 1);
    }

    #[test]
    fn no_cascade_below_threshold() {
        let mut d = CascadeDetector::new(0.95);
        let alert = d.check(100, 94); // 0.94 < 0.95
        assert!(!alert);
        assert!(!d.is_cascade_active());
    }

    #[test]
    fn recovery_detected() {
        let mut d = CascadeDetector::new(0.95);
        d.check(100, 96); // cascade
        assert!(d.is_cascade_active());
        d.check(100, 50); // recovery
        assert!(!d.is_cascade_active());
        assert_eq!(d.total_recoveries(), 1);
    }

    #[test]
    fn no_double_alert_while_active() {
        let mut d = CascadeDetector::new(0.95);
        d.check(100, 96);
        let alert2 = d.check(100, 97);
        assert!(!alert2); // already active, no new alert
        assert_eq!(d.total_cascades(), 1);
    }

    #[test]
    fn confirmation_rounds_required() {
        let mut d = CascadeDetector::new(0.95).with_confirmation_rounds(3);
        assert!(!d.check(100, 96)); // round 1
        assert!(!d.check(100, 97)); // round 2
        assert!(d.check(100, 98));  // round 3 - confirmed
    }

    #[test]
    fn confirmation_resets_on_drop_below() {
        let mut d = CascadeDetector::new(0.95).with_confirmation_rounds(3);
        d.check(100, 96); // round 1
        d.check(100, 97); // round 2
        d.check(100, 50); // drops, resets counter
        d.check(100, 96); // round 1 again
        assert!(!d.is_cascade_active());
    }

    #[test]
    fn ratio_history_tracked() {
        let mut d = CascadeDetector::new(0.95);
        d.check(100, 50);
        d.check(100, 75);
        assert_eq!(d.ratio_history().len(), 2);
        assert!((d.ratio_history()[0] - 0.5).abs() < f64::EPSILON);
        assert!((d.ratio_history()[1] - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_agents_safe() {
        let ratio = CascadeDetector::avoidance_ratio(0, 0);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut d = CascadeDetector::new(0.95);
        d.check(100, 96);
        d.reset();
        assert!(!d.is_cascade_active());
        assert_eq!(d.total_cascades(), 0);
        assert!(d.ratio_history().is_empty());
    }

    #[test]
    fn with_default_threshold() {
        let d = CascadeDetector::with_default_threshold();
        assert!((d.threshold() - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_cascade_cycles() {
        let mut d = CascadeDetector::new(0.9);
        d.check(100, 95); // cascade 1
        d.check(100, 50); // recovery 1
        d.check(100, 92); // cascade 2
        d.check(100, 60); // recovery 2
        assert_eq!(d.total_cascades(), 2);
        assert_eq!(d.total_recoveries(), 2);
    }
}
