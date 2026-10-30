import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelWordCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelLogBackward

/-! Public complete-kernel mathematical endpoints.

`Correct.forward_kernel_real_words` proves actual typed-source success, every
softmax/log output and the complete source memory frame on exact decoded finite
or negative-infinity Tensor words, with a finite effective score in each nonempty
row. `Correct.backward_kernel_real_words` proves the finite JVP, and
`Correct.log_backward_kernel_real_words` additionally allows negative-infinity
saved log values. Mathematical input domains never narrow the Python guards.

Historical wrapper endpoints are preserved intact in `Proofs/HostCorrect.lean`.
-/
