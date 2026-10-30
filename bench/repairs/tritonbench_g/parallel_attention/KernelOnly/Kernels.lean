import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.SourceSyntax
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.QuerySyntax
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.CandidateSyntax
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ShellSyntax

/-! Readable original and fused forward and complete backward source authority.
The complete shell retains the DQ/barrier/DKV order. -/
