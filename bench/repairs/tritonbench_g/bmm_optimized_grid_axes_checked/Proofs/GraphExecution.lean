import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.SourceGraph
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Complete

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Ordinary compositional proof rule; no executor is installed in the IO. -/
theorem run_congr (p : Scalar α) (rest : Algebra α) (code : List ComputeStmt)
    (steps : ∀ st∈code,∀ s : State (Node α),
      Execution.step (scalar p) (base p rest) st s=FP.Structural.step (engine p rest) st s)
    (s : State (Node α)) :
    Execution.run (scalar p) (base p rest) code s=FP.Structural.run (engine p rest) code s := by
  induction code generalizing s with
  | nil=>rw [Execution.run,FP.Structural.run]
  | cons st tail ih=>
    rw [Execution.run,FP.Structural.run,steps st (by simp)]
    cases h : FP.Structural.step (engine p rest) st s with
    | none=>rfl
    | some next=>exact ih (fun st hs=>steps st (by simp [hs])) next

/-- Exact actual loop-body correspondence for all eight source mask branches. -/
theorem body_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (s : State (Node α)) :
    Execution.run (scalar p) (base p rest) (Source.body cfg) s=
      FP.Structural.run (engine p rest) (Source.body cfg) s := by
  apply run_congr
  intro st member state
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals
    simp only [Source.body,Source.parts,Source.kernel,VeriTile.Triton.Source.dynamicRange?,
      ComputeKernel.surfaceBody,dm,dn,dk,Option.map,Option.getD,beq_self_eq_true,↓reduceIte,
      List.mem_cons,List.not_mem_nil,or_false] at member
    repeat' first
      | (rcases member with rfl | member)
      | (cases member)
    all_goals first
      | (rw [Execution.step]; exact (accumulating_step p rest _ _ state).symm)
      | (rw [Execution.step]; rfl)
    all_goals simp

/-- The source prelude contains only shared scalar/control statements. -/
theorem prelude_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (s : State (Node α)) :
    Execution.run (scalar p) (base p rest) (Source.prelude cfg) s=
      FP.Structural.run (engine p rest) (Source.prelude cfg) s := by
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals
    simp only [Source.prelude,Source.parts,Source.kernel,VeriTile.Triton.Source.dynamicRange?,
      ComputeKernel.surfaceBody,dm,dn,dk,Option.map,Option.getD,beq_self_eq_true,↓reduceIte,
      engine,Execution.run,Execution.step,Execution.localStep,FP.Structural.run,FP.Structural.step]
    rfl

/-- All source output-mask branches retain the identical shared store. -/
theorem epilogue_equal (p : Scalar α) (rest : Algebra α) (cfg : Source.Parameters)
    (s : State (Node α)) :
    Execution.run (scalar p) (base p rest) (Source.epilogue cfg) s=
      FP.Structural.run (engine p rest) (Source.epilogue cfg) s := by
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals
    simp only [Source.epilogue,Source.parts,Source.kernel,VeriTile.Triton.Source.dynamicRange?,
      ComputeKernel.surfaceBody,dm,dn,dk,Option.map,Option.getD,beq_self_eq_true,↓reduceIte,
      engine,Execution.run,Execution.step,Execution.localStep,FP.Structural.run,FP.Structural.step]
    rfl

#axiomsClean body_equal
#axiomsClean prelude_equal
#axiomsClean epilogue_equal
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
