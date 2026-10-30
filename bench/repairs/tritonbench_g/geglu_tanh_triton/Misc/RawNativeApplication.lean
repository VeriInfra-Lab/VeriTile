import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.RawNative

/-! Optional complete raw-memory native application; the main source-pair
connection remains in Proofs/RawNative. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton.RawNative
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical Native
noncomputable section

/-- Complete source-to-native correspondence over raw allocation descriptors
and physical memory. FMA/tanh, FTZ sites, constants and all FP32 words retain
the existing primitive model. Outputs and the exterior-byte frame are proved. -/
theorem geglu_kernel_fp32 {M : Type} (ops : Ops M Word (BitVec 8))
    (instructions : Instructions) (candidate backward : Bool)
    (p : Layout) (valid : Valid p) (plan : RawKernel.Plan p) (entry : M) :
    ∃ final,
      RawKernel.NativeExecutes ops instructions candidate backward p valid plan entry final ∧
      RawKernel.RunSpec ops (compiledReference instructions) backward p valid entry final := by
  refine ⟨_,RawKernel.native_executes ops instructions candidate backward p valid plan entry,?_⟩
  exact RawKernel.run_spec ops (sourceScalar (compiledPorts instructions)) (compiledReference instructions)
    (compiled_lowering instructions) backward p valid plan entry

#axiomsClean geglu_kernel_fp32
end
end VeriTile.Bench.Repairs.GegluTanhTriton.RawNative
