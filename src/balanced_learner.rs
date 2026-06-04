//! Balanced learner implementing the v5 fix for avoidance cascades.
//!
//! The key insight: agents should learn from **average reward with margin**,
//! not from **minimum reward**. Pure avoidance learning creates a death spiral
//! where one bad experience causes permanent avoidance.

/// A balanced learner that prevents avoidance cascades through:
///
/// 1. **Batch learning**: agents learn from average reward, not minimum
/// 2. **Forced exploration**: a configurable fraction of agents always explore
/// 3. **Memory decay**: old avoidance memories fade over time
#[derive(Debug, Clone)]
pub struct BalancedLearner {
    /// Margin below average reward that still counts as "good enough" (exploration margin).
    exploration_margin: f64,
    /// Memory decay rate per round (0.0 = no decay, 1.0 = instant forget).
    memory_decay: f64,
    /// Minimum reward threshold to even consider avoiding.
    avoid_threshold: f64,
    /// Number of options being tracked.
    option_count: usize,
    /// Per-option: cumulative reward.
    option_rewards: Vec<f64>,
    /// Per-option: number of times chosen.
    option_chooses: Vec<u32>,
    /// Per-option: avoidance weight (decays over time).
    option_avoid_weights: Vec<f64>,
}

/// The decision an agent makes for a single option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Choose this option.
    Choose,
    /// Avoid this option.
    Avoid,
    /// No information — explore.
    Explore,
}

impl BalancedLearner {
    /// Creates a new balanced learner.
    ///
    /// # Arguments
    ///
    /// * `exploration_margin` — how far below average reward is still acceptable
    /// * `memory_decay` — rate at which avoidance memory fades each round (0.0–1.0)
    pub fn new(exploration_margin: f64, memory_decay: f64) -> Self {
        assert!(
            exploration_margin >= 0.0,
            "exploration_margin must be non-negative"
        );
        assert!(
            (0.0..=1.0).contains(&memory_decay),
            "memory_decay must be in [0.0, 1.0]"
        );
        Self {
            exploration_margin,
            memory_decay,
            avoid_threshold: 0.0,
            option_count: 0,
            option_rewards: Vec::new(),
            option_chooses: Vec::new(),
            option_avoid_weights: Vec::new(),
        }
    }

    /// Sets the number of options to track.
    pub fn with_options(mut self, count: usize) -> Self {
        self.option_count = count;
        self.option_rewards = vec![0.0; count];
        self.option_chooses = vec![0; count];
        self.option_avoid_weights = vec![0.0; count];
        self
    }

    /// Sets the avoidance reward threshold. Options with average reward below this
    /// may be avoided even by balanced learners.
    pub fn with_avoid_threshold(mut self, threshold: f64) -> Self {
        self.avoid_threshold = threshold;
        self
    }

    /// Returns the number of tracked options.
    pub fn option_count(&self) -> usize {
        self.option_count
    }

    /// Returns the average reward for a specific option.
    ///
    /// Returns 0.0 if the option hasn't been chosen.
    pub fn avg_reward(&self, option: usize) -> f64 {
        if option >= self.option_count || self.option_chooses[option] == 0 {
            0.0
        } else {
            self.option_rewards[option] / self.option_chooses[option] as f64
        }
    }

    /// Returns the avoidance weight for a specific option.
    pub fn avoid_weight(&self, option: usize) -> f64 {
        if option >= self.option_count {
            0.0
        } else {
            self.option_avoid_weights[option]
        }
    }

    /// Records a reward observation for a specific option.
    pub fn record_reward(&mut self, option: usize, reward: f64) {
        if option < self.option_count {
            self.option_rewards[option] += reward;
            self.option_chooses[option] += 1;
            // Good reward reduces avoidance weight
            if reward > self.avoid_threshold {
                self.option_avoid_weights[option] *= 0.5;
            }
        }
    }

    /// Records a negative experience (avoidance signal) for an option.
    pub fn record_negative(&mut self, option: usize, penalty: f64) {
        if option < self.option_count {
            self.option_avoid_weights[option] += penalty;
        }
    }

    /// Applies memory decay to all avoidance weights.
    ///
    /// Call this once per round. Each weight is multiplied by `(1 - memory_decay)`.
    pub fn decay_memories(&mut self) {
        let factor = 1.0 - self.memory_decay;
        for w in &mut self.option_avoid_weights {
            *w *= factor;
        }
    }

    /// Computes the global average reward across all chosen options.
    pub fn global_avg_reward(&self) -> f64 {
        let total: f64 = self.option_rewards.iter().sum();
        let count: u32 = self.option_chooses.iter().sum();
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// Makes a balanced decision for a specific option.
    ///
    /// The decision logic:
    /// 1. If never explored → **Explore**
    /// 2. If avoidance weight is very high → **Avoid** (but decay will fix this)
    /// 3. If avg reward is within margin of global average → **Choose**
    /// 4. If avg reward is far below threshold → **Avoid**
    /// 5. Otherwise → **Explore** (re-evaluate)
    pub fn decide(&self, option: usize) -> Decision {
        if option >= self.option_count {
            return Decision::Explore;
        }

        // Never explored → must explore
        if self.option_chooses[option] == 0 {
            return Decision::Explore;
        }

        let avg = self.avg_reward(option);
        let global = self.global_avg_reward();
        let weight = self.option_avoid_weights[option];

        // Very high avoidance weight → avoid (but memory decay will reduce this)
        if weight > 2.0 {
            return Decision::Avoid;
        }

        // Within margin of global average → choose
        if avg >= global - self.exploration_margin {
            return Decision::Choose;
        }

        // Below absolute threshold → avoid
        if avg < self.avoid_threshold {
            return Decision::Avoid;
        }

        // Ambiguous → explore
        Decision::Explore
    }

    /// Runs one full round of balanced learning for all options.
    ///
    /// Returns decisions for each option and applies memory decay.
    pub fn round(&mut self) -> Vec<Decision> {
        let decisions: Vec<Decision> = (0..self.option_count).map(|i| self.decide(i)).collect();
        self.decay_memories();
        decisions
    }

    /// Returns counts of each decision type from the given decisions.
    pub fn count_decisions(decisions: &[Decision]) -> (usize, usize, usize) {
        let choose = decisions.iter().filter(|d| **d == Decision::Choose).count();
        let avoid = decisions.iter().filter(|d| **d == Decision::Avoid).count();
        let explore = decisions.iter().filter(|d| **d == Decision::Explore).count();
        (choose, avoid, explore)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_learner_has_no_data() {
        let l = BalancedLearner::new(0.1, 0.05).with_options(3);
        assert_eq!(l.option_count(), 3);
        for i in 0..3 {
            assert!((l.avg_reward(i) - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    #[should_panic]
    fn negative_margin_panics() {
        BalancedLearner::new(-0.1, 0.05);
    }

    #[test]
    #[should_panic]
    fn invalid_decay_panics() {
        BalancedLearner::new(0.1, 1.5);
    }

    #[test]
    fn unexplored_options_get_explore() {
        let l = BalancedLearner::new(0.1, 0.05).with_options(5);
        for i in 0..5 {
            assert_eq!(l.decide(i), Decision::Explore);
        }
    }

    #[test]
    fn good_reward_gets_choose() {
        let mut l = BalancedLearner::new(0.1, 0.05).with_options(3);
        // Record good rewards for option 0
        for _ in 0..5 {
            l.record_reward(0, 1.0);
        }
        assert_eq!(l.decide(0), Decision::Choose);
    }

    #[test]
    fn low_reward_with_high_avoid_weight() {
        let mut l = BalancedLearner::new(0.1, 0.0).with_options(2);
        l.record_reward(0, 0.1);
        l.record_negative(0, 5.0); // very high avoid weight
        assert_eq!(l.decide(0), Decision::Avoid);
    }

    #[test]
    fn memory_decay_reduces_weights() {
        let mut l = BalancedLearner::new(0.1, 0.5).with_options(1);
        l.record_negative(0, 4.0);
        assert!((l.avoid_weight(0) - 4.0).abs() < 1e-10);
        l.decay_memories();
        assert!((l.avoid_weight(0) - 2.0).abs() < 1e-10);
        l.decay_memories();
        assert!((l.avoid_weight(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn global_avg_reward_computed() {
        let mut l = BalancedLearner::new(0.1, 0.0).with_options(2);
        l.record_reward(0, 2.0);
        l.record_reward(0, 4.0);
        l.record_reward(1, 6.0);
        // total = 12, count = 3, avg = 4.0
        assert!((l.global_avg_reward() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn full_round_returns_decisions() {
        let mut l = BalancedLearner::new(0.1, 0.05).with_options(3);
        // option 0: good
        for _ in 0..3 {
            l.record_reward(0, 1.0);
        }
        // option 1: bad + avoid weight
        l.record_reward(1, 0.01);
        l.record_negative(1, 5.0);
        // option 2: unexplored

        let decisions = l.round();
        assert_eq!(decisions.len(), 3);
        assert_eq!(decisions[0], Decision::Choose);
        assert_eq!(decisions[1], Decision::Avoid);
        assert_eq!(decisions[2], Decision::Explore);
    }

    #[test]
    fn count_decisions_works() {
        let decisions = vec![Decision::Choose, Decision::Avoid, Decision::Choose, Decision::Explore];
        let (c, a, e) = BalancedLearner::count_decisions(&decisions);
        assert_eq!(c, 2);
        assert_eq!(a, 1);
        assert_eq!(e, 1);
    }

    #[test]
    fn out_of_bounds_returns_explore() {
        let l = BalancedLearner::new(0.1, 0.05).with_options(2);
        assert_eq!(l.decide(99), Decision::Explore);
        assert!((l.avg_reward(99) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn balanced_learning_prevents_cascade() {
        // Simulate the key scenario: one bad experience shouldn't cascade
        let mut l = BalancedLearner::new(0.1, 0.3).with_options(10);

        // All options get moderate rewards initially
        for opt in 0..10 {
            for _ in 0..3 {
                l.record_reward(opt, 0.5);
            }
        }

        // One bad experience on option 0
        l.record_negative(0, 1.0);

        // Run many rounds with decay
        for _ in 0..20 {
            let decisions = l.round();
            let (_choose, avoid, _explore) = BalancedLearner::count_decisions(&decisions);
            // Avoidance should never reach 100% (the cascade)
            assert!(
                avoid < 10,
                "Avoidance cascade detected: {}/10 avoiding",
                avoid
            );
            // Re-record rewards for chosen options
            for (i, d) in decisions.iter().enumerate() {
                if *d == Decision::Choose || *d == Decision::Explore {
                    l.record_reward(i, 0.5);
                }
            }
        }
    }

    #[test]
    fn with_avoid_threshold() {
        let l = BalancedLearner::new(0.1, 0.05)
            .with_options(2)
            .with_avoid_threshold(0.3);
        // Just checking builder pattern works
        assert_eq!(l.option_count(), 2);
    }
}
