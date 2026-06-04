# avoidance-cascade

A Rust crate that models, detects, and prevents the **avoidance cascade** phenomenon in ternary agent systems.

## The Problem

In a **ternary action space** (choose / avoid / unknown), agents that learn purely from avoidance signals converge to a pathological state: **they avoid everything** (100% avoid, 0% choose). This is the **avoidance cascade** — a death spiral where one bad experience cascades into total avoidance.

### How It Happens

1. Agent tries option A → gets negative reward → marks A as "avoid"
2. Agent tries option B → gets negative reward → marks B as "avoid"
3. ... eventually all options are marked "avoid"
4. Agent never explores again → 100% avoid, 0% choose

This is especially dangerous in batch/minibatch learning where the **minimum reward** in a batch dominates the signal.

## The Fix (v5 Balanced Learning)

This crate implements the **v5 fix** — balanced batch learning with three mechanisms:

### 1. Average Reward (Not Minimum)
Agents learn from the **average reward** across a batch, not the minimum. One bad experience in a batch of 10 doesn't dominate.

### 2. Forced Exploration
An exploration scheduler ensures agents periodically re-evaluate "avoided" options. The exploration rate decays over time but never drops below a configurable floor.

### 3. Memory Decay
Avoidance memories fade over time. An option avoided 100 rounds ago shouldn't carry the same weight as one avoided just now.

## Usage

```rust
use avoidance_cascade::{CascadeDetector, BalancedLearner, ExplorationScheduler, CascadeMetrics};

// Detect cascades
let mut detector = CascadeDetector::new(0.95); // alert when 95%+ agents avoid

// Balanced learning prevents cascades
let mut learner = BalancedLearner::new(0.1, 0.05) // margin=0.1, decay=0.05
    .with_options(10);

// Force exploration
let mut scheduler = ExplorationScheduler::new(0.1) // 10% decay per round
    .with_min_rate(0.05); // never below 5%

// Track metrics
let mut metrics = CascadeMetrics::new();

// Run simulation rounds
for round in 0..100 {
    let decisions = learner.round();
    let (choose, avoid, explore) = BalancedLearner::count_decisions(&decisions);
    metrics.record(choose, avoid, explore);

    if detector.check(choose + avoid + explore, avoid) {
        metrics.record_cascade();
        println!("Cascade detected at round {round}!");
    }
}
```

## Components

| Component | Description |
|-----------|-------------|
| `CascadeDetector` | Monitors avoidance ratio, raises alerts when it exceeds threshold |
| `BalancedLearner` | Prevents cascades via average-reward learning + forced exploration + memory decay |
| `ExplorationScheduler` | Forces periodic exploration with decaying rate and configurable floor |
| `CascadeMetrics` | Tracks ratios, cascade events, and recovery events over time |
| `TernaryAction` | The three-way action enum (Choose, Avoid, Unknown) |
| `AgentOption` | Per-option state tracking for individual agents |

## Installation

```toml
[dependencies]
avoidance-cascade = "0.1.0"
```

## License

MIT
