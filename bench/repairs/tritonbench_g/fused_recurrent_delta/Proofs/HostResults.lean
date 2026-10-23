import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.BackwardTiles
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.RealSum
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
noncomputable section
open Real
open scoped BigOperators
set_option maxHeartbeats 12000000
namespace RS
open VeriTile.Bench.Repairs.MatmulKernel.RealSum
/-- Tiled host summation covers every feature exactly once, with masked tails. -/
theorem tile_total (s : Shape) (a : Nat → ℝ) :
    (∑ iv∈Finset.range s.BNV,tileSum s iv a)=∑ v : Fin s.V,a v.val := by
  have chunks_fin (n : Nat) : (∑ iv∈Finset.range n,tileSum s iv a)=
      chunks (fun v => if v<s.V then a v else 0) s.BBV n := by
    induction n with
    | zero => simp [chunks]
    | succ n ih =>
      rw [Finset.sum_range_succ,ih,chunks,sum_finset]
      rfl
  rw [chunks_fin,chunks_eq,sum_padding]
  have hc : s.V≤s.BNV*s.BBV := ceil_cover s.V s.BBV (Geometry.tile_pos s).2.2
  rw [Nat.min_eq_left hc,sum_finset,←Fin.sum_univ_eq_sum_range]
end RS

theorem host_q {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (b t k : Nat) (hb : b<s.BH) (ht : t<s.T) (hk : k<s.K) :
    hostReduce s f p g m ⟨.retq,b,t,k⟩=p.cut ((g.host s.BNV).eval p (fun iv => m ⟨.dq,iv*s.BH+b,t,k⟩)) := by
  simp [hostReduce,write,hb,ht,hk]
theorem host_k {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (b t k : Nat) (hb : b<s.BH) (ht : t<s.T) (hk : k<s.K) :
    hostReduce s f p g m ⟨.retk,b,t,k⟩=p.cut ((g.hostK s.BNV).eval p (fun iv => m ⟨.dk,iv*s.BH+b,t,k⟩)) := by
  simp [hostReduce,write,hb,ht,hk]
theorem host_v {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (b t v : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : v<s.V) :
    hostReduce s f p g m ⟨.retv,b,t,v⟩=p.cut ((g.hostV 1).eval p (fun _ => m ⟨.dv,b,t,v⟩)) := by
  simp [hostReduce,write,hb,ht,hv]
theorem host_beta {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m : Memory α)
    (b t v : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : v<(if f.vectorBeta then s.V else 1)) :
    hostReduce s f p g m ⟨.retbeta,b,t,v⟩=p.cut ((g.hostBeta s.BNV).eval p (fun iv => m ⟨.dbeta,iv*s.BH+b,t,v⟩)) := by
  simp [hostReduce,write,hb,ht,hv]

def finish (s : Shape) (f : Flags) (gf gb : Trees) (scale : ℝ) (m : Memory ℝ) :=
  hostReduce s f ops gb (backwardGrid s f ops gb scale (saved s f gf scale m))
structure Legal (s : Shape) (gf gb : Trees) : Prop where
  fkey : (gf.key s.BK).Legal s.BK
  freadout : (gf.readout s.BK).Legal s.BK
  bkey : (gb.key s.BK).Legal s.BK
  bvalue : (gb.value s.BBV).Legal s.BBV
  beta : (gb.beta s.BBV).Legal s.BBV
  correction : (gb.correction s.BBV).Legal s.BBV
  query : (gb.query s.BBV).Legal s.BBV
  hostQ : (gb.host s.BNV).Legal s.BNV
  hostK : (gb.hostK s.BNV).Legal s.BNV
  hostV : (gb.hostV 1).Legal 1
  hostBeta : (gb.hostBeta s.BNV).Legal s.BNV

theorem return_q (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b t : Nat) (i : Fin s.K) (hb : b<s.BH) (ht : t<s.T) :
    finish s f gf gb scale m ⟨.retq,b,t,i.val⟩=∑ v : Fin s.V,(grad s f scale m b v.val t).q i := by
  rw [finish,host_q s f ops gb _ b t i.val hb ht i.isLt]
  change (gb.host s.BNV).eval ops _=_
  rw [tree_sum _ _ hl.hostQ,←RS.tile_total s (fun v => (grad s f scale m b v t).q i)]
  apply Finset.sum_congr rfl
  intro iv hi
  rw [BackwardResults.dq_value s f ops gb hl.correction scale _ b iv t i.val hb (Finset.mem_range.mp hi) ht i.isLt]
  exact query_tile s f gf gb hl.fkey hl.query scale m b iv t i hb ht

theorem return_k (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b t : Nat) (i : Fin s.K) (hb : b<s.BH) (ht : t<s.T) :
    finish s f gf gb scale m ⟨.retk,b,t,i.val⟩=∑ v : Fin s.V,(grad s f scale m b v.val t).k i := by
  rw [finish,host_k s f ops gb _ b t i.val hb ht i.isLt]
  change (gb.hostK s.BNV).eval ops _=_
  rw [tree_sum _ _ hl.hostK,←RS.tile_total s (fun v => (grad s f scale m b v t).k i)]
  apply Finset.sum_congr rfl
  intro iv hi
  rw [BackwardResults.dk_value s f ops gb hl.correction scale _ b iv t i.val hb (Finset.mem_range.mp hi) ht i.isLt]
  exact key_tile s f gf gb hl.fkey hl.bkey hl.bvalue hl.correction scale m b iv t i hb ht

theorem return_v (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b t v : Nat) (hb : b<s.BH) (ht : t<s.T) (hv : v<s.V) :
    finish s f gf gb scale m ⟨.retv,b,t,v⟩=(grad s f scale m b v t).x := by
  obtain ⟨ho,hj,he⟩:=Geometry.owner_exists s.V s.BBV v (Geometry.tile_pos s).2.2 hv
  rw [finish,host_v s f ops gb _ b t v hb ht hv]
  change (gb.hostV 1).eval ops _=_
  rw [tree_sum _ _ hl.hostV]
  simp only [Finset.sum_range_one]
  have hr:=BackwardResults.dv_value s f ops gb hl.correction scale (saved s f gf scale m) b (v/s.BBV) t (v%s.BBV) hb ho ht hj (by simpa [he] using hv)
  rw [he] at hr
  rw [hr]
  change Reverse.dv s f ops gb scale _ b (v/s.BBV) t (v%s.BBV)=_
  rw [dv s f gf gb hl.bkey scale m b (v/s.BBV) (v%s.BBV) t (by simpa [he] using hv),he]

theorem return_beta_scalar (s : Shape) (f : Flags) (gf gb : Trees) (hl : Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) (b t : Nat) (hb : b<s.BH) (ht : t<s.T) (hf : f.vectorBeta=Bool.false) :
    finish s f gf gb scale m ⟨.retbeta,b,t,0⟩=∑ v : Fin s.V,(grad s f scale m b v.val t).beta := by
  rw [finish,host_beta s f ops gb _ b t 0 hb ht (by simp [hf])]
  change (gb.hostBeta s.BNV).eval ops _=_
  rw [tree_sum _ _ hl.hostBeta,←RS.tile_total s (fun v => (grad s f scale m b v t).beta)]
  apply Finset.sum_congr rfl
  intro iv hi
  rw [BackwardResults.dbeta_scalar_value s f ops gb hl.correction scale _ b iv t hb (Finset.mem_range.mp hi) ht hf]
  exact beta_tile s f gf gb hl.fkey hl.bkey hl.beta scale m b iv t hb ht

#axiomsClean RS.tile_total
#axiomsClean return_q
#axiomsClean return_k
#axiomsClean return_v
#axiomsClean return_beta_scalar
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RealBackward
