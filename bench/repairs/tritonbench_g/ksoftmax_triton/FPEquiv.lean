import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostFPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelNative
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelSchedule

/-! Public complete-kernel native endpoints.

`FPEquiv.forward_kernel_native` and `backward_kernel_native` prove actual source
success and every output word/byte under the selected local compiler/ABI model.
The input state comes from the original Tensor pointers, strides, storage offsets
and one common initial word/four-byte map. All FP32 patterns are admitted and
both source memory and exterior bytes are framed. `FPEquiv.empty_kernel` needs no
compiled object or pointer/numerical domain. `KernelSchedule` executes a reversed
complete source schedule and proves equality of the entire source memory.

Historical endpoints and the existing eight deferred instruction-local numerical
ports are preserved intact in `Proofs/HostFPEquiv.lean` and its dependencies.
-/
