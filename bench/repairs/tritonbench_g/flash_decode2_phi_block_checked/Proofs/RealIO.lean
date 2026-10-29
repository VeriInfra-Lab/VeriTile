import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.RealIO
open VeriTile.Triton FP.Structural FlashDecode2 PhiIO
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

def observe (x : Option ℝ) : ℝ := x.getD 0

theorem value (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (xs : PhiIO.Inputs ℝ g) (domain : PhiIO.Fits g xs ∧ PhiIO.Positive g xs)
    (s : State (Option ℝ))
    (pins : (PhiIO.decodeIO Reference.ideal g order some observe).InputBinding xs s)
    (row : Fin (g.B*g.H)) (d : Fin g.D) :
    observe (Grid.value Reference.ideal s Bool.true g.B g.H g.S g.D g.Q g.BD row.val
      (⟨d.val,lt_of_lt_of_le d.isLt width⟩,PUnit.unit))=
      Reference.target (PhiIO.logs xs row.val) (PhiIO.mids xs row.val) (PhiIO.steps xs row.val) d.val := by
  have counted := count_binding Reference.ideal g order some observe xs s pins row.val row.isLt
  let entry := Grid.setup g.B g.H row.val s
  have hx : ∀ j,j<count entry g.Q → logvalue Reference.ideal entry g.H g.S j=
      some (PhiIO.logs xs row.val j) := by
    intro j hj
    have used : j<PhiIO.steps xs row.val := counted ▸ hj
    simp only [logvalue,entry,row_log_offset,Reference.ideal]
    exact log_binding Reference.ideal g order some observe xs s pins row.val j row.isLt
      (lt_of_lt_of_le used (domain.1 row.val row.isLt)) used
  have hv : ∀ j,j<count entry g.Q → ∀ k : TileIndex [g.BD],
      FlashDecode2.value Reference.ideal entry Bool.true g.H g.S g.D j k=
      some (PhiIO.mids xs row.val j (lane k)) := by
    intro j hj k
    have used : j<PhiIO.steps xs row.val := counted ▸ hj
    by_cases active : lane k<g.D
    · simp only [FlashDecode2.value,FlashDecode2.active,Bool.true_eq_false,false_or,active,if_pos,
        entry,row_mid_offset,Reference.ideal]
      exact mid_binding Reference.ideal g order some observe xs s pins row.val j (lane k)
        row.isLt (lt_of_lt_of_le used (domain.1 row.val row.isLt)) used active
    · simp [FlashDecode2.value,FlashDecode2.active,active,zero,Reference.ideal,PhiIO.mids]
      rfl
  have output := Reference.source_correct entry Bool.true g.H g.S g.D g.Q g.BD
    (PhiIO.logs xs row.val) (fun j k => PhiIO.mids xs row.val j (lane k))
    (counted.symm ▸ domain.2 row.val row.isLt) hx hv
    (⟨d.val,lt_of_lt_of_le d.isLt width⟩,PUnit.unit)
  change observe (FlashDecode2.output Reference.ideal entry Bool.true g.H g.S g.D g.Q g.BD _) = _
  rw [output]
  change Reference.target _ _ (count entry g.Q) _ = _
  rw [counted]
  rfl

theorem implements (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    (PhiIO.decodeIO Reference.ideal g order some observe).ImplementsOn
      (fun xs => PhiIO.Fits g xs ∧ PhiIO.Positive g xs) (PhiIO.mathematical g) :=
  PhiIO.implements Reference.ideal g width order coverage some observe
    _ (fun _ h => h.1) _ (value g width order)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.RealIO
