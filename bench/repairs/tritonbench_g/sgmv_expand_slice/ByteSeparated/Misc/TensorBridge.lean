import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Correct
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.NativeCorrect
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Real
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Access
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.OutputFrame
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Guard

/-! Optional caller-Tensor and byte observations of the native kernel result. -/
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.TensorBridge
open VeriTile.Triton FP.Structural Contract
noncomputable section

/-- Executable admission checks imply complete typed source execution and the
independent ordinary K-term formula in original segment order. -/
theorem checked_real (t : Inputs) (s : Scalars) (lists : Lists t) (ok : Admitted t s spans lists)
    (rest : Algebra ℝ) (X : Nat→Nat→ℝ) (W : Nat→Nat→Nat→ℝ) (entry : Nat→Nat→ℝ) :
    ∃ final,
      runSourceJobs (SourceBody.install (RealBackend.algebra rest) RealBackend.dot)
        (abi t s) (ByteSeparated.args t s) (jobs (shape t s).columns (segments t s lists))
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat))=some final ∧
      SourceSuffix.OutputMemory (abi t s) (ByteSeparated.args t s) (shape t s) final
        (mathematicalResult (shape t s) X W s.add (segments t s lists) entry) ∧
      SourceGrid.ReadFrame (ByteSeparated.args t s)
        (PreparedMemory.state (abi t s) (shape t s) X W entry (raw t s lists)
          (lists.lengths.drop s.batches.toNat)) final :=
  ByteSeparated.Real.complete t s lists ok rest X W entry

abbrev nativeReference := reference

theorem checked_fp (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (zeros : NativeBackend.ZeroConversions ops) (t : Inputs) (s : Scalars)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) :
    execution rest ops t s before=some (finalState rest ops t s before) ∧
    SourceSuffix.OutputMemory (abi t s) (args t s) (shape t s) (finalState rest ops t s before)
      (nativeReference ops t s before) ∧
    SourceGrid.ReadFrame (args t s) (initial t s before) (finalState rest ops t s before) :=
  complete rest ops zeros t s before ok

/-- Exact candidate source identity and retained individual conversion/add/
m16n8k16 instruction agreements. No whole-output or source-success premise. -/
structure CompilerBinding (referenceOps : Native.Primitives) where
  sourceAST : String
  sourceIdentity : sourceAST=productionAST
  primitives : Native.Primitives
  agreement : Native.PrimitiveAgreement primitives referenceOps

/-- An observation of the in-place call, preserving the actual supplied core
Tensor object and None return; bytes are the completed source's own stores. -/
structure Result where
  tensor : VeriTile.Torch.FP32.Tensor
  returned : Unit
  bytes : Raw.State

def invoke (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans) (before : Raw.State) : Result :=
  ⟨a.output.tensor,(),sourceStores rest ops t s before⟩

theorem checked_caller_bytes (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (compiler : CompilerBinding ops) (zeros : NativeBackend.ZeroConversions ops)
    (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) :
    let result:=invoke rest compiler.primitives t s spans a before
    result.tensor=a.output.tensor ∧ result.returned=() ∧
    execution rest compiler.primitives t s before=some (finalState rest compiler.primitives t s before) ∧
    (∀ cell,Touched (shape t s) (segments t s (lists t before)) cell → ∀ byte : Fin 4,
      result.bytes t.y.device.index ((key t s cell).address+byte.val)=
        Raw.unpack (nativeReference ops t s before cell.1 cell.2) byte) ∧
    (∀ owner address,ReadByte spans address → result.bytes owner address=before owner address) ∧
    (∀ owner address,(∀ cell,Touched (shape t s) (segments t s (lists t before)) cell →
      ¬(key t s cell).covers owner address) → result.bytes owner address=before owner address) := by
  rw [Native.primitives_ext compiler.primitives ops compiler.agreement]
  exact ⟨rfl,rfl,(whole rest ops zeros t s before ok).1,
    (whole rest ops zeros t s before ok).2.1,
    caller_frame rest ops t s spans a before ok,source_frame rest ops t s before⟩

#axiomsClean checked_real
#axiomsClean checked_fp
#specNonCircular nativeReference avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#axiomsClean checked_caller_bytes
#specNonCircular CompilerBinding avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.TensorBridge
