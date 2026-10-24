import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyForward
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyBackward

/-! Actual source safety for the complete forward and backward signed loops.
The per-statement transitions are kept separate to avoid duplicating execution
proofs or unfolding floating predecessor states in the public entries. -/
