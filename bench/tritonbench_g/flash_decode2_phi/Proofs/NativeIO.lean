import bench.tritonbench_g.flash_decode2_phi.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.NativeIO
open VeriTile.Triton FP.Structural FlashDecode2 PhiIO
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

theorem term_binding (g : PhiIO.Geometry) (order : List Nat)
    (xs : PhiIO.Inputs Native.Term g) (fits : PhiIO.Fits g xs) (s : State Native.Term)
    (pins : (PhiIO.decodeIO Native.algebra g order id id).InputBinding xs s)
    (row : Fin (g.B*g.H)) (d : Nat) :
    Raw.nativeTerm s g.B g.H g.S g.D g.Q row.val d=PhiIO.nativeTerm g xs row.val d := by
  have counted := count_binding Native.algebra g order id id xs s pins row.val row.isLt
  have hx (j : Nat) (used : j<PhiIO.steps xs row.val) :
      Raw.logs s g.S row.val j=Native.Term.load (PhiIO.logs xs row.val j) := by
    unfold Raw.logs
    rw [log_binding Native.algebra g order id id xs s pins row.val j row.isLt
      (lt_of_lt_of_le used (fits row.val row.isLt)) used]
    rfl
  have hv (j : Nat) (used : j<PhiIO.steps xs row.val) :
      Raw.mids s g.S g.D row.val j=(fun d => Native.Term.load
        (if d<g.D then PhiIO.mids xs row.val j d else .literal (some .fp32) .real 0)) := by
    funext d
    unfold Raw.mids
    by_cases live : d<g.D
    · rw [if_pos live,if_pos live]
      rw [mid_binding Native.algebra g order id id xs s pins row.val j d row.isLt
        (lt_of_lt_of_le used (fits row.val row.isLt)) used live]
      rfl
    · simp only [if_neg live]
  have scan (n : Nat) (hn : n≤PhiIO.steps xs row.val) :
      KernelNative.scan (Raw.logs s g.S row.val) (Raw.mids s g.S g.D row.val) n=
      KernelNative.scan (fun j => Native.Term.load (PhiIO.logs xs row.val j))
        (fun j d => Native.Term.load
          (if d<g.D then PhiIO.mids xs row.val j d else .literal (some .fp32) .real 0)) n := by
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [KernelNative.scan,KernelNative.scan,ih (by omega),hx n (by omega),hv n (by omega)]
  unfold Raw.nativeTerm PhiIO.nativeTerm
  rw [counted,scan _ le_rfl]

theorem value (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (xs : PhiIO.Inputs Native.Term g) (fits : PhiIO.Fits g xs) (s : State Native.Term)
    (pins : (PhiIO.decodeIO Native.algebra g order id id).InputBinding xs s)
    (row : Fin (g.B*g.H)) (d : Fin g.D) :
    id (Grid.value Native.algebra s Bool.true g.B g.H g.S g.D g.Q g.BD row.val
      (⟨d.val,lt_of_lt_of_le d.isLt width⟩,PUnit.unit))=PhiIO.nativeTerm g xs row.val d.val := by
  rw [id_eq,Raw.native_value]
  exact term_binding g order xs fits s pins row d.val

theorem implements (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    (PhiIO.decodeIO Native.algebra g order id id).ImplementsOn (PhiIO.Fits g)
      (fun xs => PhiIO.result (fun row d => PhiIO.nativeTerm g xs row.val d.val)) :=
  PhiIO.implements Native.algebra g width order coverage id id _ (fun _ h => h)
    _ (value g width order)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.NativeIO
