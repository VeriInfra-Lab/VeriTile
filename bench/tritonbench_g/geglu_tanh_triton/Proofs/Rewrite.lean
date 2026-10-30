import bench.tritonbench_g.geglu_tanh_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Rewrite
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- Only the signed integer tile-offset product differs. The complete scalar
suffix, including its native FMA/tanh interpretation, is literally unchanged. -/
theorem prefix_eq {α : Type} [Inhabited α] (M : Algebra α)
    (backward : Bool) (block count : Nat) (s : State α) :
    run M ((source backward block count).surfaceBody.take 2) s =
      run M ((indexCommutedSource backward block count).surfaceBody.take 2) s := by
  have product :
      bop (fun a b : Int => a*b) .nil
          (fun _ : TileIndex [] => (s.pids 0 : Int)) (fun _ => (block : Int)) =
        bop (fun a b : Int => a*b) .nil
          (fun _ : TileIndex [] => (block : Int)) (fun _ => (s.pids 0 : Int)) := by
    funext i
    exact Int.mul_comm _ _
  cases backward <;>
    simp [source,indexCommutedSource,forwardKernel,backwardKernel,
      forwardIndexCommuted,backwardIndexCommuted,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalOp_unfold,numeric,State.setReg,product]

theorem suffix_eq (backward : Bool) (block count : Nat) :
    (source backward block count).surfaceBody.drop 2 =
      (indexCommutedSource backward block count).surfaceBody.drop 2 := by
  cases backward <;> rfl

theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α)
    (backward : Bool) (block count : Nat) (s : State α) :
    run M (source backward block count).surfaceBody s =
      run M (indexCommutedSource backward block count).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 2 (source backward block count).surfaceBody,run_append]
  conv_rhs => rw [←List.take_append_drop 2 (indexCommutedSource backward block count).surfaceBody,run_append]
  rw [prefix_eq,suffix_eq]

theorem selected_run {α : Type} [Inhabited α] (M : Algebra α)
    (backward candidate : Bool) (block count : Nat) (s : State α) :
    run M (selectedSource backward candidate block count).surfaceBody s =
      run M (source backward block count).surfaceBody s := by
  cases candidate
  · rfl
  · exact (execution_eq M backward block count s).symm

#axiomsClean execution_eq
#axiomsClean selected_run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Rewrite
