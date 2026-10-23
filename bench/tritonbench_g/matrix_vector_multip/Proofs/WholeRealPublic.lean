import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeReal
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeContract

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
open VeriTile VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost WholeFinite
noncomputable section
set_option maxHeartbeats 6000000

variable {α : Type} {x : Inputs} {entry : Nat → α} {preexisting : Nat → Prop}

theorem real_outcomes {entry : Nat → ℝ} (rt : Runtime ℝ x entry preexisting) :
    Outcomes (fun _ i => WholeReal.mathematical x entry i) rt (execute WholeReal.family rt) := by
  by_cases compatible : x.columns=x.vectorLength
  · cases allocated : rt.allocate with
    | none => simp [execute,compatible,allocated,Outcomes]
    | some a =>
      cases ready : rt.launchReady a with
      | «false» =>
        simp [execute,compatible,allocated,ready,Outcomes]
      | «true» =>
        obtain ⟨out,ran,shape,dtype,device,stride,base,values,exterior,frame⟩ :=
          WholeReal.execute_returns x entry preexisting rt compatible a allocated ready
        rw [ran]
        exact ⟨compatible,a,allocated,ready,shape,dtype,device,stride,base,values,exterior,frame⟩
  · simp [execute,compatible,Outcomes]

/-- Independent standard matrix-vector formula, complete original source and
host, every autotune trial, exhaustive outcomes and full physical boundaries. -/
specification mv_real_source_outcomes {entry : Nat → ℝ}
    (rt : Runtime ℝ x entry preexisting)
    (finite : ∀ a,rt.allocate=some a → rt.launchReady a=Bool.true → FiniteAdmission x entry preexisting a) :
    Spec.Real (Outcomes (fun _ i => WholeReal.mathematical x entry i) rt (execute WholeReal.family rt) ∧
      ∀ a,rt.allocate=some a → rt.launchReady a=Bool.true → IndexCorrespondence a) := by
  exact ⟨real_outcomes rt,fun a allocated ready => index_correspondence a (finite a allocated ready)⟩


#axiomsClean mv_real_source_outcomes

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
