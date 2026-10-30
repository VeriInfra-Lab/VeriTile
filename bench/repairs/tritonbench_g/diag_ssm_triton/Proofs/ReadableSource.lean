import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceRealRun
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceRealSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardRun
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexRun
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardRun
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourcePair

/-! The copied Python bodies retain signed casts and their own temporary
registers. The imported execution and access-safety theorems connect their
complete runs to the independently proved mathematical implementations.
There is no assertion that the new and historical ASTs or register states are
identical. SourcePair separately compares the actual original/candidate bodies. -/
