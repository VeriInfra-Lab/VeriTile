import bench.repairs.tritonbench_g.fused_recurrent_delta.Correct
import bench.repairs.tritonbench_g.fused_recurrent_delta.FPEquiv

/-! Kernel-only public bundle. Per-program source execution, shared Tensor
interpretation, grid coverage and separation constitute the kernel proof.
Historical wrapper/autograd models are available explicitly in Misc/HostWhole.lean. -/
