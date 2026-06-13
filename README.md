# avoidance-cascade

**Rust library for detecting and preventing avoidance cascades in ternary agent systems (+1 choose, 0 unknown, -1 avoid).**

In a ternary action space, agents that learn purely from minimum-reward signals converge to **avoiding everything** — a degenerate equilibrium where the avoidance ratio reaches 100% and the system produces zero reward. This is the **avoidance cascade**: a self-reinforcing spiral where one bad experience permanently biases an agent against an option, and this bias propagates through shared learning. This crate provides four integrated tools: cascade detection, balanced learning, exploration scheduling, and metrics tracking.

## Why It Matters

Avoidance cascades are a known failure mode in multi-agent systems, particularly in:

- **Multi-agent RL** with avoidance actions — agents learn to avoid all options, producing zero-reward episodes indefinitely.
- **Recommendation systems** — "avoid" signals from a few users can cascade through collaborative filtering, permanently suppressing content.
- **Organizational decision-making** — risk-avoidance contagion where one team's failure causes adjacent teams to refuse similar work.
- **Autonomous systems** — robots that learn to avoid all navigation paths after one collision.

The mathematical structure is that of an **information cascade** (Bikhchandani et al., 1992): once enough agents avoid, the rational inference for new agents is to also avoid, regardless of their private information. The result is **herding on the worst outcome**.

## How It Works

### Cascade Detection

`CascadeDetector` monitors the avoidance ratio ρ = avoid_count / total_agents. A cascade is declared when:

> ρ ≥ threshold  for  `confirmation_rounds` consecutive rounds

Default threshold: 0.95 (95% of agents avoiding). The confirmation window prevents false alarms from transient spikes.

State machine: INACTIVE → (ρ ≥ threshold for R rounds) → ACTIVE → (ρ < threshold) → RECOVERED.

### Balanced Learning (v5 Fix)

The core insight: **agents should learn from average reward, not minimum reward.** Pure avoidance learning (min-reward) creates a death spiral because min() is monotonically non-increasing — one bad experience permanently lowers the floor.

`BalancedLearner` implements three anti-cascade mechanisms:

**1. Exploration margin**: An option is "chooseable" if its average reward r̄(i) satisfies:

> r̄(i) ≥ global_avg - margin

where global_avg = Σ r̄(i) / k (mean across all options). This means an option only gets avoided if it's significantly worse than the population average, not just worse than the best option.

**2. Forced exploration**: Unknown options (0 observations) always get `Decision::Explore`.

**3. Memory decay**: Avoidance weights decay each round:

> w_i(t+1) = w_i(t) · (1 - decay_rate)

With decay_rate = 0.05, an avoidance weight of 1.0 decays to ~0.60 after 10 rounds and ~0.13 after 40 — bad memories fade.

### Decision Logic

```
decide(option i):
  if never explored → Explore
  if avoid_weight[i] > high_threshold → Avoid
  if avg_reward[i] ≥ global_avg - margin → Choose
  if avg_reward[i] < avoid_threshold → Avoid
  else → Explore (re-evaluate)
```

### Exploration Scheduler

`ExplorationScheduler` decays exploration rate ε over rounds:

> ε(t) = max(ε₀ · decay^t, ε_min)

Starting at ε₀ = 1.0 (all explore), decaying to ε_min = 0.05 (5% always explore). The floor ensures the system never fully converges — there's always some exploration to break out of local optima.

### Metrics

`CascadeMetrics` tracks snapshots over time: avoidance_ratio, exploration_count, is_cascading. Supports queries: peak_avoidance_ratio, mean_avoidance_ratio, cascade_events (steps where cascade started).

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `CascadeDetector::check()` | O(1) | O(H) for history |
| `BalancedLearner::decide(i)` | O(1) | O(k) for k options |
| `BalancedLearner::decay_memories()` | O(k) | O(1) in-place |
| `BalancedLearner::round()` | O(k) | O(k) |
| `ExplorationScheduler::advance()` | O(1) | O(1) |
| `CascadeMetrics::record()` | O(1) | O(1) amortized |

## Quick Start

```rust
use avoidance_cascade::{CascadeDetector, BalancedLearner, ExplorationScheduler, Decision};

// Detect cascades
let mut detector = CascadeDetector::new(0.95);
let alert = detector.check(100, 96); // 96/100 avoiding
assert!(alert);
assert!(detector.is_cascade_active());

// Balanced learning prevents cascade
let mut learner = BalancedLearner::new(0.1, 0.3) // margin=0.1, decay=0.3
    .with_options(10);

// All options get moderate rewards
for opt in 0..10 {
    for _ in 0..3 { learner.record_reward(opt, 0.5); }
}
// One bad experience
learner.record_negative(0, 1.0);

// Run rounds — avoidance never reaches 100%
for _ in 0..20 {
    let decisions = learner.round();
    let (_, avoid, _) = BalancedLearner::count_decisions(&decisions);
    assert!(avoid < 10, "Cascade detected: {}/10 avoiding", avoid);
    learner.decay_memories();
}
```

## API

- **`CascadeDetector`** — Threshold-based cascade monitor: `check(total, avoid) → bool`, `with_confirmation_rounds(n)`
- **`BalancedLearner`** — Anti-cascade decision-maker: `decide(i) → Decision`, `round() → Vec<Decision>`, `record_reward()`, `record_negative()`, `decay_memories()`
- **`Decision`** — Choose, Avoid, Explore
- **`ExplorationScheduler`** — Decaying ε schedule: `should_explore(agent, pop) → bool`, `forced_explorers(pop) → Vec<usize>`, `advance()`
- **`CascadeMetrics`** — Time-series tracking: `record()`, `peak_avoidance_ratio`, `cascade_events()`
- **`MetricsSnapshot`** — { step, avoidance_ratio, exploration_count, is_cascading }
- **`TernaryAction`** — Choose, Avoid, Unknown
- **`AgentOption`** — { choose_count, avoid_count, total_reward, is_avoiding }

## Architecture Notes

The γ+η=C identity: γ (generative capacity) is the diversity of agent decisions — a healthy system has agents choosing, avoiding, and exploring different options. η (evaluative depth) is the learning algorithm's ability to distinguish genuine avoidance from cascaded avoidance. When η is high (balanced learner + decay), γ stays high (diverse actions). When η fails (pure min-reward learning), γ → 0 (all avoid). C = system reward throughput = f(γ, η).

## References

1. Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). "A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades." *Journal of Political Economy*, 100(5).
2. Sutton, R. & Barto, A. (2018). *Reinforcement Learning* (2nd ed.), §2.2 on ε-greedy exploration.
3. Easley, D. & Kleinberg, J. (2010). *Networks, Crowds, and Markets*, Ch. 16. — Information cascades.
4. Achlioptas, D. (2001). "Lower Bounds for Random-3SAT via Differential Equations." — Phase transitions in threshold phenomena.

## License

MIT
