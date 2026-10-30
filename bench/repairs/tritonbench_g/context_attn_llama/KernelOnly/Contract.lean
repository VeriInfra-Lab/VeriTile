import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Tensor

/-! Public contract entry: actual ordinary CUDA FP32 Tensor/storage API laws,
Four INT32 vectors and the page table from the common original byte heap, and the unchanged checked
source guards. The output is the supplied Tensor. Read roles may alias.
No execution, allocator, compiler, host-success or result premise is accepted. -/

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
#specNonCircular Contract avoiding [VeriTile.Triton.exec,VeriTile.Triton.FP.Structural.exec]
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
