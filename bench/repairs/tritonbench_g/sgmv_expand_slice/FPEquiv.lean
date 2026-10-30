import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Compiler

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.FPEquiv
open VeriTile.Triton FP.Structural Contract Direct Direct.NativeCall
noncomputable section
abbrev nativeReference := Direct.NativeCall.reference

theorem checked_fp (rest : Algebra Raw.Word) (ops : Native.Primitives) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before)) :
    execution rest ops t s a before=some (finalState rest ops t s a before) ∧
    SourceSuffix.OutputMemory (abi t s) (Direct.args t s) (shape t s) (finalState rest ops t s a before)
      (nativeReference ops t s a before) ∧
    SourceGrid.ReadFrame (Direct.args t s) (initial t s a before) (finalState rest ops t s a before) :=
  complete rest ops zeros t s a before ok

/-- Exact caller bytes from the production source and local native ports.
The supplied Y identity, return value and all bytes outside stores are retained. -/
theorem checked_caller_bytes (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (binding : CompilerBinding ops) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before)) :
    let result:=invoke rest binding.primitives t s a before ok
    result.tensor=t.y ∧ result.returned=() ∧
    execution rest binding.primitives t s a before=some (finalState rest binding.primitives t s a before) ∧
    (∀ cell,Touched (shape t s) (segments t s (lists a before)) cell → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((key a s cell).address+byte.val)=
        Raw.unpack (nativeReference ops t s a before cell.1 cell.2) byte) ∧
    (∀ owner address,(∀ cell,Touched (shape t s) (segments t s (lists a before)) cell →
      ¬(key a s cell).covers owner address) → result.bytes owner address=before owner address) := by
  exact ⟨rfl,rfl,compiled rest ops binding zeros t s a before ok⟩

#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#specNonCircular nativeReference avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.FPEquiv
