import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedExecution
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def program (p : Config) (BD : Nat) (r : Bool) : Program :=
  (if r then FPEquivProofs.reciprocal else FPEquivProofs.repaired)
    p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim 128 64 BD p.stage p.validStage

theorem program_source (p : Config) (BD : Nat) (r : Bool) :
    (program p BD r).kernel=source p BD r := by cases r <;> rfl

/-- The actual complete state is shared by two successful source executions.
Only the displayed final scalar reciprocal rule and actual operand guards
are numerical premises; dot support has no numerical equation. -/
theorem successful_equivalence {α : Type} [Inhabited α] (R : FPEquivProofs.Rules)
    (A : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions A D)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (s : State α)
    (hd : ∀ site∈(program p BD Bool.false).domain, site.Holds A D s) :
    ∃ t, exec (engine A) (source p BD Bool.false) s=some t ∧
      exec (engine A) (source p BD Bool.true) s=some t := by
  obtain ⟨t,ht⟩ := success A p BD support Bool.false s
  have he := FP.ProfiledRewrite.sound (FPEquivProofs.attention_reciprocal_equiv
    p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim 128 64 BD p.stage p.validStage R) α A D hM s hd
  change exec (engine A) (program p BD Bool.false).kernel s =
    exec (engine A) (program p BD Bool.true).kernel s at he
  rw [program_source,program_source] at he
  exact ⟨t,ht,he.symm.trans ht⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
