import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.BackwardResults
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.ForwardResults
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RealReverse
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
noncomputable section
open Real
set_option maxHeartbeats 8000000
/-- Forward saved memory, including the repaired fresh dbeta allocation. The
argument m is after default-beta preparation and before the v overwrite. -/
def saved (s : Shape) (f : Flags) (gf : Trees) (scale : ℝ) (m : Memory ℝ) :=
  prepareDbeta s f ops (forwardGrid s f ops gf scale m)
def Stable : Region → Prop
  | .q | .k | .beta | .h0 | .do_ | .dht => True
  | _ => False
theorem saved_stable (s : Shape) (f : Flags) (gf : Trees) (scale : ℝ) (m : Memory ℝ)
    (r : Region) (hr : Stable r) (b t j : Nat) :
    saved s f gf scale m ⟨r,b,t,j⟩=m ⟨r,b,t,j⟩ := by
  have h:=MemoryProof.forward_grid_frame s f ops gf scale m ⟨r,b,t,j⟩
  have hn : ¬MemoryProof.ForwardWrites s f ⟨r,b,t,j⟩ := by
    cases r <;> simp_all [Stable,MemoryProof.ForwardWrites]
  have hn' : r≠.dbeta := by cases r <;> simp_all [Stable]
  simp only [saved,prepareDbeta]
  split <;> simp [write,hn',h hn]
theorem saved_residual (s : Shape) (f : Flags) (gf : Trees) (hg : (gf.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (b t v : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : v<s.V) :
    saved s f gf scale m ⟨.v,b,t,v⟩=
      Math.residual (Math.state (column s f m b v) t) ((column s f m b v).k t) ((column s f m b v).x t) := by
  have he : saved s f gf scale m ⟨.v,b,t,v⟩=forwardGrid s f ops gf scale m ⟨.v,b,t,v⟩ := by
    simp only [saved,prepareDbeta]
    split <;> simp [write]
  rw [he,ForwardResults.residual_real s f gf hg scale m b t v hb ht hv]

theorem mid_readonly (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (bh iv : Nat) (r : Region) (hr : BackwardRelation.ReadOnly r) (b t j : Nat) :
    BackwardResults.mid s f ops g scale m bh iv ⟨r,b,t,j⟩=m ⟨r,b,t,j⟩ := by
  cases r <;> simp_all [BackwardRelation.ReadOnly,BackwardResults.mid,stateStore,write,Reverse.memory,ite_apply]
theorem mid_dk (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv t : Nat) (i : Fin s.K) (ht : t<s.T) :
    BackwardResults.mid s f ops g scale m b iv ⟨.dk,iv*s.BH+b,t,i.val⟩=
      Reverse.dk s f ops g scale m b iv t i.val := by
  simp [BackwardResults.mid,stateStore,write,Reverse.memory,ite_apply,ht,i.isLt,ops]
theorem mid_dv (s : Shape) (f : Flags) (g : Trees) (scale : ℝ) (m : Memory ℝ)
    (b iv t j : Nat) (ht : t<s.T) (hj : j<s.BBV) (hv : iv*s.BBV+j<s.V) :
    BackwardResults.mid s f ops g scale m b iv ⟨.dv,b,t,iv*s.BBV+j⟩=
      Reverse.dv s f ops g scale m b iv t j := by
  have hh : iv*s.BBV+j<(iv+1)*s.BBV := by rw [Nat.add_mul];omega
  simp [BackwardResults.mid,stateStore,write,Reverse.memory,ite_apply,ht,hv,hh,ops]

theorem saved_adjoint (s : Shape) (f : Flags) (gf : Trees) (scale : ℝ) (m : Memory ℝ)
    (bh v n : Nat) :
    adjoint s f scale (saved s f gf scale m) bh v n=adjoint s f scale m bh v n := by
  have hq : (column s f (saved s f gf scale m) bh v).q=(column s f m bh v).q := by
    funext t i;exact saved_stable s f gf scale m .q (by trivial) _ _ _
  have hk : (column s f (saved s f gf scale m) bh v).k=(column s f m bh v).k := by
    funext t i;exact saved_stable s f gf scale m .k (by trivial) _ _ _
  have hb : (column s f (saved s f gf scale m) bh v).beta=(column s f m bh v).beta := by
    funext t
    simp only [column,saved_stable s f gf scale m .beta (by trivial)]
  have hi : incoming (saved s f gf scale m) bh v=incoming m bh v := by
    funext t;exact saved_stable s f gf scale m .do_ (by trivial) _ _ _
  have ht : terminal s f (saved s f gf scale m) bh v=terminal s f m bh v := by
    funext i;simp only [terminal,saved_stable s f gf scale m .dht (by trivial)]
  unfold adjoint
  rw [hi,ht]
  exact adjoint_primal_independent _ _ _ _ _ _ _ hq hk hb

theorem saved_enhanced (s : Shape) (f : Flags) (gf : Trees) (scale : ℝ) (m : Memory ℝ)
    (bh v t : Nat) :
    enhanced s f scale (saved s f gf scale m) bh v t=enhanced s f scale m bh v t := by
  funext i
  simp only [enhanced,saved_adjoint,column,incoming,saved_stable s f gf scale m .q (by trivial),
    saved_stable s f gf scale m .do_ (by trivial)]
/-- Recompute h is proved from the overwritten residual tensor, including
its actual prior stores, not from a supplied primal state sequence. -/
theorem recompute_h (s : Shape) (f : Flags) (gf gb : Trees) (hg : (gf.key s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (b iv j : Nat) (hb : b<s.BH) (hv : iv*s.BBV+j<s.V)
    (n : Nat) (hn : n≤s.T) (i : Fin s.K) :
    Recompute.h s f ops (BackwardResults.mid s f ops gb scale (saved s f gf scale m) b iv) b iv n i.val j=
      Math.state (column s f m b (iv*s.BBV+j)) n i := by
  induction n with
  | zero =>
    simp only [Recompute.h,seed,load,i.isLt,hv,and_self,if_true]
    rw [mid_readonly s f gb scale _ b iv .h0 (by trivial),saved_stable s f gf scale m .h0 (by trivial)]
    simp [Math.state,column,ops]
  | succ n ih =>
    rw [Recompute.h]
    change Recompute.h s f ops _ b iv n i.val j+
      keyLoad s ops _ .k b n i.val*(valLoad s ops _ .v b n iv s.BBV j*betaLoad s f ops _ b n iv s.BBV j)=_
    rw [ih (by omega)]
    simp only [keyLoad,valLoad,betaLoad,load,i.isLt,hv,if_true,
      mid_readonly s f gb scale _ b iv .k (by trivial),mid_readonly s f gb scale _ b iv .v (by trivial),
      mid_readonly s f gb scale _ b iv .beta (by trivial),
      saved_stable s f gf scale m .k (by trivial),saved_stable s f gf scale m .beta (by trivial),
      saved_residual s f gf hg scale m b n (iv*s.BBV+j) hb (by omega) hv]
    simp only [Math.state,Math.next,column]
    ring

#axiomsClean saved_residual
#axiomsClean mid_dk
#axiomsClean mid_dv
#axiomsClean saved_adjoint
#axiomsClean recompute_h
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
