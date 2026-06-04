//! # Avoidance Cascade
//!
//! Models the **avoidance cascade** phenomenon from ternary agent systems where
//! pure avoidance learning leads to "avoid everything" (100% avoid, 0% choose).
//!
//! In a ternary action space (`choose`, `avoid`, `unknown`), agents that learn
//! purely from avoidance signals converge to avoiding every option. This crate
//! provides tools to:
//!
//! - **Detect** cascades as they form ([`CascadeDetector`])
//! - **Prevent** cascades via balanced learning ([`BalancedLearner`])
//! - **Force exploration** on a schedule ([`ExplorationScheduler`])
//! - **Track metrics** over time ([`CascadeMetrics`])
//!
//! ## Quick Example
//!
//! ```
//! use avoidance_cascade::{CascadeDetector, BalancedLearner, ExplorationScheduler};
//!
//! let mut detector = CascadeDetector::new(0.95);
//! let mut learner = BalancedLearner::new(0.1, 0.05);
//! let mut scheduler = ExplorationScheduler::new(0.1);
//!
//! // Simulate a round of decisions
//! let agent_count = 100;
//! let avoid_count = 96;
//!
//! assert!(detector.check(agent_count, avoid_count));
//! ```

mod cascade_detector;
mod balanced_learner;
mod exploration_scheduler;
mod metrics;

pub use cascade_detector::CascadeDetector;
pub use balanced_learner::BalancedLearner;
pub use metrics::{CascadeMetrics, MetricsSnapshot};
pub use exploration_scheduler::ExplorationScheduler;

/// The three possible actions in a ternary agent system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryAction {
    /// Agent actively chooses this option.
    Choose,
    /// Agent actively avoids this option.
    Avoid,
    /// Agent has no information about this option.
    Unknown,
}

impl TernaryAction {
    /// Returns `true` if this is a [`Choose`](TernaryAction::Choose) action.
    pub fn is_choose(&self) -> bool {
        matches!(self, TernaryAction::Choose)
    }

    /// Returns `true` if this is an [`Avoid`](TernaryAction::Avoid) action.
    pub fn is_avoid(&self) -> bool {
        matches!(self, TernaryAction::Avoid)
    }

    /// Returns `true` if this is an [`Unknown`](TernaryAction::Unknown) action.
    pub fn is_unknown(&self) -> bool {
        matches!(self, TernaryAction::Unknown)
    }
}

/// A single agent's state tracking an option.
#[derive(Debug, Clone)]
pub struct AgentOption {
    /// Number of times the agent chose this option.
    pub choose_count: u32,
    /// Number of times the agent avoided this option.
    pub avoid_count: u32,
    /// Total reward accumulated from choosing this option.
    pub total_reward: f64,
    /// Whether the agent is currently avoiding this option.
    pub is_avoiding: bool,
}

impl AgentOption {
    /// Creates a new, unexplored option.
    pub fn new() -> Self {
        Self {
            choose_count: 0,
            avoid_count: 0,
            total_reward: 0.0,
            is_avoiding: false,
        }
    }

    /// Returns the current action the agent would take.
    pub fn action(&self) -> TernaryAction {
        if self.is_avoiding {
            TernaryAction::Avoid
        } else if self.choose_count > 0 {
            TernaryAction::Choose
        } else {
            TernaryAction::Unknown
        }
    }

    /// Returns the average reward for this option.
    pub fn avg_reward(&self) -> f64 {
        if self.choose_count == 0 {
            0.0
        } else {
            self.total_reward / self.choose_count as f64
        }
    }

    /// Records a choice with the given reward.
    pub fn record_choose(&mut self, reward: f64) {
        self.choose_count += 1;
        self.total_reward += reward;
        self.is_avoiding = false;
    }

    /// Records an avoidance event.
    pub fn record_avoid(&mut self) {
        self.avoid_count += 1;
        self.is_avoiding = true;
    }

    /// Resets the avoidance flag (for exploration).
    pub fn clear_avoidance(&mut self) {
        self.is_avoiding = false;
    }
}

impl Default for AgentOption {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_action_predicates() {
        assert!(TernaryAction::Choose.is_choose());
        assert!(!TernaryAction::Choose.is_avoid());
        assert!(TernaryAction::Avoid.is_avoid());
        assert!(TernaryAction::Unknown.is_unknown());
    }

    #[test]
    fn agent_option_starts_unknown() {
        let opt = AgentOption::new();
        assert_eq!(opt.action(), TernaryAction::Unknown);
        assert_eq!(opt.choose_count, 0);
        assert_eq!(opt.avoid_count, 0);
        assert_eq!(opt.avg_reward(), 0.0);
    }

    #[test]
    fn agent_option_choose_transitions() {
        let mut opt = AgentOption::new();
        opt.record_choose(1.0);
        assert_eq!(opt.action(), TernaryAction::Choose);
        assert_eq!(opt.choose_count, 1);
        assert!((opt.avg_reward() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn agent_option_avoid_transitions() {
        let mut opt = AgentOption::new();
        opt.record_avoid();
        assert_eq!(opt.action(), TernaryAction::Avoid);
        assert_eq!(opt.avoid_count, 1);
    }

    #[test]
    fn agent_option_clear_avoidance() {
        let mut opt = AgentOption::new();
        opt.record_avoid();
        assert!(opt.is_avoiding);
        opt.clear_avoidance();
        assert!(!opt.is_avoiding);
        // Still unknown since no chooses recorded
        assert_eq!(opt.action(), TernaryAction::Unknown);
    }

    #[test]
    fn agent_option_avg_reward_multiple() {
        let mut opt = AgentOption::new();
        opt.record_choose(2.0);
        opt.record_choose(4.0);
        assert!((opt.avg_reward() - 3.0).abs() < f64::EPSILON);
    }
}
