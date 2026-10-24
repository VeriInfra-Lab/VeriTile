import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.NativeReference
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.NativePublicIO

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeCorrect
open VeriTile Triton FP.Structural
noncomputable section

/-- The runtime FP32 epsilon remains a register argument. Both full grids keep
all FP32 accumulation, reductions, source operation order and FP16 round trip. -/
abbrev nativeIO := NativeIO.original

/-- Native word formula and source/candidate equivalence for the original
0 < columns ≤ 16384 domain. Arbitrary symbolic inputs remain admitted. The
primitive instructions, reduction topology and lowering schedule are explicit
model data; this integer-only rewrite adds no numerical transformation law. -/
theorem rmsnorm_kernel_native (I : Native.Instructions) (R : Native.Reductions)
    (S : Native.Schedule) (X Y W : RegionName) (rows columns : Nat) (epsilon : Native.Word)
    (positive : 0<columns) (limit : columns≤16384) (xy : X≠Y) (wy : W≠Y) :
    (nativeIO R X Y W rows columns epsilon).Observes (fun _ _ => Native.eval I S)
      (Reference.native I R S X Y W rows columns epsilon) ∧
    (nativeIO R X Y W rows columns epsilon).ObservedEquivalent
      (NativeIO.commuted (Native.algebra R) X Y W rows columns 16384 (.leaf "epsilon" 0 epsilon))
      (fun _ _ => Native.eval I S) := by
  exact ⟨NativePublicIO.native I R S X Y W rows columns epsilon positive limit xy wy,
    ProgramIO.ObservedEquivalent.of_equivalent
      (NativePublicIO.equivalent (Native.algebra R) X Y W rows columns 16384
        (.leaf "epsilon" 0 epsilon) positive limit xy wy) _⟩

#axiomsClean rmsnorm_kernel_native
#specNonCircular RuntimeReference.native avoiding [runtimeKernel,FP.Structural.exec,RuntimeRow.value,RuntimeGrid.graph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeCorrect
