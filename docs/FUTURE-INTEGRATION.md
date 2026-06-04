# Future Integration: avoidance-cascade

## Current State
Models the avoidance cascade phenomenon where pure avoidance learning leads to "avoid everything" (100% avoid, 0% choose). Provides `CascadeDetector` (threshold-based detection), `BalancedLearner` (avoid:choose ratio enforcement), `ExplorationScheduler` (forced exploration), and `CascadeMetrics` with `MetricsSnapshot` tracking.

## Integration Opportunities

### With Cell Population Death Spiral Prevention
In a `CellGrid` tissue, if too many cells emit `TernaryMessenger::Suppress`, the tissue enters an avoidance cascade — cells stop communicating, the tissue dies. `CascadeDetector::check()` monitors the tissue's suppress ratio. When it exceeds 0.95, `ExplorationScheduler` forces cells to emit Signal messages, breaking the cascade. `BalancedLearner` maintains the tissue's communication health.

### With ternary-scheduling-v2
`CascadeDetector` prevents a scheduling death spiral where `Priority::Reject` cascades — once jobs start being rejected, the system rejects more aggressively until nothing runs. `BalancedLearner` ensures a minimum fraction of jobs get `Priority::Normal` or `Priority::Priority`. `ExplorationScheduler` periodically admits speculative jobs to prevent the rejection cascade.

### With negative-space-core
The cascade is the extreme case of negative-space-core's 294:1 ratio. At 100% avoidance, the ratio goes to infinity — the conservation law breaks. `CascadeDetector` is the safety valve that prevents the conservation law from being violated. `MetricsSnapshot` tracks cascade proximity, giving early warning before the 294:1 ratio degrades.

## Potential in Mature Systems
Cascade prevention is critical at every level. Room cascades: rooms stop communicating, fleet fragments. Agent cascades: agents stop exploring, fitness stagnates. Cell cascades: tissues die. At each level, `CascadeDetector` monitors, `BalancedLearner` enforces balance, and `ExplorationScheduler` forces diversity. This is the immune system against monoculture death.

## Cross-Pollination Ideas
- `CascadeMetrics` could feed into `ternary-entropy` — cascade onset is measurable as entropy collapse
- `BalancedLearner`'s ratio enforcement connects to `conservation-matrix-rs` — the learner maintains the conservation law
- `ExplorationScheduler` could use `ternary-fitness` landscape topology to guide exploration toward high-value regions
- `MetricsSnapshot` over time creates a cascade early-warning dataset for `ternary-science` analysis

## Dependencies for Next Steps
- Integration with ternary-cell's tissue coordination for real-time cascade monitoring
- Threshold calibration from live fleet data (the 0.95 default may not be universal)
- Integration with ternary-room for room-level cascade detection
- Connect to ternary-scheduling-v2 for job-level cascade prevention
