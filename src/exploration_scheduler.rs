//! Exploration scheduler that forces agents to periodically explore unknown options.

/// Forces agents to periodically explore (choose unknown) based on a configurable
/// exploration probability that decays over time.
///
/// The exploration rate starts at `initial_rate` and decays by `decay_rate` each
/// round, with a configurable floor to ensure some exploration always occurs.
#[derive(Debug, Clone)]
pub struct ExplorationScheduler {
    /// Initial exploration probability.
    initial_rate: f64,
    /// Decay factor applied each round (e.g., 0.95 means 5% decay).
    decay_rate: f64,
    /// Minimum exploration rate (floor).
    min_rate: f64,
    /// Current exploration rate.
    current_rate: f64,
    /// Current round number.
    round: u64,
    /// Total forced explorations.
    total_forced: u64,
}

impl ExplorationScheduler {
    /// Creates a new scheduler with the given decay rate.
    ///
    /// The initial exploration rate is 1.0 (always explore).
    ///
    /// # Panics
    ///
    /// Panics if `decay_rate` is not in `(0.0, 1.0]`.
    pub fn new(decay_rate: f64) -> Self {
        assert!(
            decay_rate > 0.0 && decay_rate <= 1.0,
            "decay_rate must be in (0.0, 1.0]"
        );
        Self {
            initial_rate: 1.0,
            decay_rate,
            min_rate: 0.05,
            current_rate: 1.0,
            round: 0,
            total_forced: 0,
        }
    }

    /// Sets the initial exploration rate.
    pub fn with_initial_rate(mut self, rate: f64) -> Self {
        assert!((0.0..=1.0).contains(&rate), "initial_rate must be in [0.0, 1.0]");
        self.initial_rate = rate;
        self.current_rate = rate;
        self
    }

    /// Sets the minimum exploration rate floor.
    pub fn with_min_rate(mut self, rate: f64) -> Self {
        assert!((0.0..=1.0).contains(&rate), "min_rate must be in [0.0, 1.0]");
        self.min_rate = rate;
        self
    }

    /// Returns the current exploration rate.
    pub fn current_rate(&self) -> f64 {
        self.current_rate
    }

    /// Returns the current round number.
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Returns total number of forced explorations.
    pub fn total_forced(&self) -> u64 {
        self.total_forced
    }

    /// Determines whether an agent should be forced to explore this round.
    ///
    /// Uses a simple deterministic schedule: if `agent_id % population_size`
    /// falls within the exploration fraction, that agent explores.
    pub fn should_explore(&self, agent_id: usize, population_size: usize) -> bool {
        if population_size == 0 {
            return false;
        }
        let explore_count = (self.current_rate * population_size as f64).ceil() as usize;
        let explore_count = explore_count.max(1).min(population_size);
        agent_id < explore_count
    }

    /// Advances to the next round, decaying the exploration rate.
    pub fn advance(&mut self) {
        self.round += 1;
        self.current_rate = (self.current_rate * self.decay_rate).max(self.min_rate);
    }

    /// Forces exploration for a subset of agents and returns their IDs.
    ///
    /// Returns a vector of agent indices that should explore this round.
    pub fn forced_explorers(&mut self, population_size: usize) -> Vec<usize> {
        let explore_count = (self.current_rate * population_size as f64).ceil() as usize;
        let explore_count = explore_count.max(1).min(population_size);
        self.total_forced += explore_count as u64;
        (0..explore_count).collect()
    }

    /// Resets the scheduler to initial state.
    pub fn reset(&mut self) {
        self.current_rate = self.initial_rate;
        self.round = 0;
        self.total_forced = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scheduler_starts_at_one() {
        let s = ExplorationScheduler::new(0.95);
        assert!((s.current_rate() - 1.0).abs() < f64::EPSILON);
        assert_eq!(s.round(), 0);
    }

    #[test]
    #[should_panic]
    fn zero_decay_panics() {
        ExplorationScheduler::new(0.0);
    }

    #[test]
    #[should_panic]
    fn over_one_decay_panics() {
        ExplorationScheduler::new(1.5);
    }

    #[test]
    fn decay_reduces_rate() {
        let mut s = ExplorationScheduler::new(0.9);
        s.advance();
        assert!((s.current_rate() - 0.9).abs() < 1e-10);
        s.advance();
        assert!((s.current_rate() - 0.81).abs() < 1e-10);
    }

    #[test]
    fn rate_never_goes_below_min() {
        let mut s = ExplorationScheduler::new(0.1).with_min_rate(0.2);
        for _ in 0..100 {
            s.advance();
        }
        assert!(s.current_rate() >= 0.2 - f64::EPSILON);
    }

    #[test]
    fn should_explore_distribution() {
        let s = ExplorationScheduler::new(0.95).with_initial_rate(0.5);
        let pop = 10;
        // 50% of 10 = 5 agents explore
        let exploring: Vec<usize> = (0..pop).filter(|&i| s.should_explore(i, pop)).collect();
        assert_eq!(exploring, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn forced_explorers_returns_subset() {
        let mut s = ExplorationScheduler::new(0.95).with_initial_rate(0.3);
        let explorers = s.forced_explorers(10);
        // ceil(0.3 * 10) = 3
        assert_eq!(explorers, vec![0, 1, 2]);
        assert_eq!(s.total_forced(), 3);
    }

    #[test]
    fn zero_population_safe() {
        let s = ExplorationScheduler::new(0.9);
        assert!(!s.should_explore(0, 0));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = ExplorationScheduler::new(0.9);
        s.advance();
        s.advance();
        s.forced_explorers(10);
        s.reset();
        assert!((s.current_rate() - 1.0).abs() < f64::EPSILON);
        assert_eq!(s.round(), 0);
        assert_eq!(s.total_forced(), 0);
    }

    #[test]
    fn custom_initial_rate() {
        let s = ExplorationScheduler::new(0.9).with_initial_rate(0.5);
        assert!((s.current_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn decay_converges_to_min() {
        let mut s = ExplorationScheduler::new(0.5).with_min_rate(0.1);
        for _ in 0..50 {
            s.advance();
        }
        assert!((s.current_rate() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn forced_explorers_always_at_least_one() {
        let mut s = ExplorationScheduler::new(0.95).with_initial_rate(0.001);
        let explorers = s.forced_explorers(100);
        assert!(!explorers.is_empty());
    }
}
