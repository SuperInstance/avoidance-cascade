//! Metrics tracking for cascade monitoring.

/// A snapshot of agent population metrics at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSnapshot {
    /// Fraction of agents choosing.
    pub choose_ratio: f64,
    /// Fraction of agents avoiding.
    pub avoid_ratio: f64,
    /// Fraction of agents in unknown state.
    pub unknown_ratio: f64,
    /// The round this snapshot was taken.
    pub round: u64,
}

impl MetricsSnapshot {
    /// Creates a new snapshot from raw counts.
    pub fn from_counts(round: u64, choose: usize, avoid: usize, unknown: usize) -> Self {
        let total = choose + avoid + unknown;
        if total == 0 {
            return Self {
                choose_ratio: 0.0,
                avoid_ratio: 0.0,
                unknown_ratio: 0.0,
                round,
            };
        }
        let t = total as f64;
        Self {
            choose_ratio: choose as f64 / t,
            avoid_ratio: avoid as f64 / t,
            unknown_ratio: unknown as f64 / t,
            round,
        }
    }
}

/// Tracks avoidance ratio, choose ratio, unknown ratio, cascade events,
/// and recovery events over time.
#[derive(Debug, Clone)]
pub struct CascadeMetrics {
    /// All recorded snapshots.
    snapshots: Vec<MetricsSnapshot>,
    /// Total cascade events.
    cascade_events: u64,
    /// Total recovery events.
    recovery_events: u64,
    /// Current round.
    round: u64,
}

impl CascadeMetrics {
    /// Creates empty metrics.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            cascade_events: 0,
            recovery_events: 0,
            round: 0,
        }
    }

    /// Records a new snapshot from action counts.
    pub fn record(&mut self, choose: usize, avoid: usize, unknown: usize) {
        let snapshot = MetricsSnapshot::from_counts(self.round, choose, avoid, unknown);
        self.snapshots.push(snapshot);
        self.round += 1;
    }

    /// Records a cascade event.
    pub fn record_cascade(&mut self) {
        self.cascade_events += 1;
    }

    /// Records a recovery event.
    pub fn record_recovery(&mut self) {
        self.recovery_events += 1;
    }

    /// Returns all recorded snapshots.
    pub fn snapshots(&self) -> &[MetricsSnapshot] {
        &self.snapshots
    }

    /// Returns the latest snapshot, if any.
    pub fn latest(&self) -> Option<&MetricsSnapshot> {
        self.snapshots.last()
    }

    /// Returns the total cascade events.
    pub fn cascade_events(&self) -> u64 {
        self.cascade_events
    }

    /// Returns the total recovery events.
    pub fn recovery_events(&self) -> u64 {
        self.recovery_events
    }

    /// Returns the current round.
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Computes the average avoid ratio across all snapshots.
    pub fn avg_avoid_ratio(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.avoid_ratio).sum();
        sum / self.snapshots.len() as f64
    }

    /// Computes the average choose ratio across all snapshots.
    pub fn avg_choose_ratio(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.choose_ratio).sum();
        sum / self.snapshots.len() as f64
    }

    /// Computes the maximum avoid ratio seen.
    pub fn max_avoid_ratio(&self) -> f64 {
        self.snapshots
            .iter()
            .map(|s| s.avoid_ratio)
            .fold(0.0, f64::max)
    }

    /// Computes the minimum choose ratio seen.
    pub fn min_choose_ratio(&self) -> f64 {
        self.snapshots
            .iter()
            .map(|s| s.choose_ratio)
            .fold(1.0, f64::min)
    }

    /// Resets all metrics.
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.cascade_events = 0;
        self.recovery_events = 0;
        self.round = 0;
    }
}

impl Default for CascadeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_metrics() {
        let m = CascadeMetrics::new();
        assert!(m.snapshots().is_empty());
        assert!(m.latest().is_none());
        assert_eq!(m.cascade_events(), 0);
        assert_eq!(m.recovery_events(), 0);
        assert!((m.avg_avoid_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_snapshot() {
        let mut m = CascadeMetrics::new();
        m.record(30, 60, 10);
        let latest = m.latest().unwrap();
        assert!((latest.choose_ratio - 0.3).abs() < 1e-10);
        assert!((latest.avoid_ratio - 0.6).abs() < 1e-10);
        assert!((latest.unknown_ratio - 0.1).abs() < 1e-10);
        assert_eq!(latest.round, 0);
        assert_eq!(m.round(), 1);
    }

    #[test]
    fn zero_counts_safe() {
        let snap = MetricsSnapshot::from_counts(0, 0, 0, 0);
        assert!((snap.choose_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cascade_and_recovery_events() {
        let mut m = CascadeMetrics::new();
        m.record_cascade();
        m.record_cascade();
        m.record_recovery();
        assert_eq!(m.cascade_events(), 2);
        assert_eq!(m.recovery_events(), 1);
    }

    #[test]
    fn avg_avoid_ratio() {
        let mut m = CascadeMetrics::new();
        m.record(50, 50, 0);   // 0.5
        m.record(80, 20, 0);   // 0.2
        assert!((m.avg_avoid_ratio() - 0.35).abs() < 1e-10);
    }

    #[test]
    fn avg_choose_ratio() {
        let mut m = CascadeMetrics::new();
        m.record(30, 70, 0);
        m.record(70, 30, 0);
        assert!((m.avg_choose_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn max_avoid_ratio() {
        let mut m = CascadeMetrics::new();
        m.record(10, 80, 10); // 0.8
        m.record(50, 30, 20); // 0.3
        m.record(5, 95, 0);   // 0.95
        assert!((m.max_avoid_ratio() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn min_choose_ratio() {
        let mut m = CascadeMetrics::new();
        m.record(10, 80, 10); // 0.1
        m.record(50, 30, 20); // 0.5
        assert!((m.min_choose_ratio() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_all() {
        let mut m = CascadeMetrics::new();
        m.record(50, 50, 0);
        m.record_cascade();
        m.reset();
        assert!(m.snapshots().is_empty());
        assert_eq!(m.cascade_events(), 0);
        assert_eq!(m.round(), 0);
    }

    #[test]
    fn default_trait() {
        let m = CascadeMetrics::default();
        assert_eq!(m.round(), 0);
    }

    #[test]
    fn multiple_rounds_increment() {
        let mut m = CascadeMetrics::new();
        for _ in 0..5 {
            m.record(50, 50, 0);
        }
        assert_eq!(m.snapshots().len(), 5);
        assert_eq!(m.round(), 5);
        // Verify rounds are numbered correctly
        assert_eq!(m.snapshots()[0].round, 0);
        assert_eq!(m.snapshots()[4].round, 4);
    }

    #[test]
    fn snapshot_equality() {
        let s1 = MetricsSnapshot::from_counts(0, 50, 50, 0);
        let s2 = MetricsSnapshot::from_counts(0, 50, 50, 0);
        assert_eq!(s1, s2);
    }
}
