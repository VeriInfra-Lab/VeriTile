import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.StagedArithmetic

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1500000

theorem dot_congr (a : Arithmetic α) (p : DotPlan) (x y x' y' : Nat → α)
    (hx : ∀ i, i∈p.leaves → x i=x' i) (hy : ∀ i, i∈p.leaves → y i=y' i) :
    p.eval a x y=p.eval a x' y' := by
  induction p with
  | product i => simp [DotPlan.eval,hx i (by simp [DotPlan.leaves]),hy i (by simp [DotPlan.leaves])]
  | add l r hl hr =>
    simp only [DotPlan.eval]
    rw [hl (fun i hi => hx i (by simp [DotPlan.leaves,hi]))
      (fun i hi => hy i (by simp [DotPlan.leaves,hi])),
      hr (fun i hi => hx i (by simp [DotPlan.leaves,hi]))
      (fun i hi => hy i (by simp [DotPlan.leaves,hi]))]
  | fma i t ht =>
    simp only [DotPlan.eval]
    rw [hx i (by simp [DotPlan.leaves]),hy i (by simp [DotPlan.leaves]),
      ht (fun j hj => hx j (by simp [DotPlan.leaves,hj]))
      (fun j hj => hy j (by simp [DotPlan.leaves,hj]))]

theorem bounded_reduction (a : Arithmetic α) (p : Plans) (axno n : Nat) (hn : 0<n)
    (kind : Kind) (x y : Nat → α) (covers : (choosePlan p axno kind).Covers n) :
    reduceNodes a p axno (fun j => if j<n then product a kind (x j) (y j) else atom a.zero) =
      atom ((choosePlan p axno kind).eval a x y) := by
  simp only [reduceNodes,hn,if_true,product]
  congr 1
  apply dot_congr
  all_goals
    intro i hi
    have bound : i<n := List.mem_range.mp (covers.mem_iff.mp hi)
    simp [bound,product]

theorem output_reduction (a : Arithmetic α) (p : Plans) (BK BV : Nat)
    (hBK : 0<BK) (covered : p.output.Covers BK) (x : Nat → Nat → α) (y : Nat → α) :
    sumValues a p (shape := [BV,BK]) ⟨1,by simp⟩ Bool.false
      (fun i => product a .outputDot (x i.1.val i.2.1.val) (y i.2.1.val)) =
      fun i => atom (p.output.eval a (x i.1.val) y) := by
  funext i
  change reduceNodes a p 1 _ = _
  simpa [TileShape.axisDim,TileShape.insertAxisIndex,choosePlan] using
    bounded_reduction a p 1 BK hBK .outputDot (x i.1.val) y covered

theorem query_reduction (a : Arithmetic α) (p : Plans) (BK BV : Nat)
    (hBV : 0<BV) (covered : p.dq.Covers BV) (x : Nat → Nat → α) (y : Nat → α) :
    sumValues a p (shape := [BK,BV]) ⟨1,by simp⟩ Bool.false
      (fun i => product a .queryDot (x i.1.val i.2.1.val) (y i.2.1.val)) =
      fun i => atom (p.dq.eval a (x i.1.val) y) := by
  funext i
  change reduceNodes a p 1 _ = _
  simpa [TileShape.axisDim,TileShape.insertAxisIndex,choosePlan] using
    bounded_reduction a p 1 BV hBV .queryDot (x i.1.val) y covered

theorem key_reduction (a : Arithmetic α) (p : Plans) (BK BV : Nat)
    (hBV : 0<BV) (covered : p.dk.Covers BV) (x : Nat → Nat → α) (y : Nat → α) :
    sumValues a p (shape := [BK,BV]) ⟨1,by simp⟩ Bool.false
      (fun i => product a .reverseDot (x i.1.val i.2.1.val) (y i.2.1.val)) =
      fun i => atom (p.dk.eval a (x i.1.val) y) := by
  funext i
  change reduceNodes a p 1 _ = _
  simpa [TileShape.axisDim,TileShape.insertAxisIndex,choosePlan] using
    bounded_reduction a p 1 BV hBV .reverseDot (x i.1.val) y covered

theorem value_reduction (a : Arithmetic α) (p : Plans) (BK BV : Nat)
    (hBK : 0<BK) (covered : p.dv.Covers BK) (x : Nat → Nat → α) (y : Nat → α) :
    sumValues a p (shape := [BK,BV]) ⟨0,by simp⟩ Bool.false
      (fun i => product a .reverseDot (x i.1.val i.2.1.val) (y i.1.val)) =
      fun i => atom (p.dv.eval a (fun k => x k i.1.val) y) := by
  funext i
  change reduceNodes a p 0 _ = _
  simpa [TileShape.axisDim,TileShape.insertAxisIndex,choosePlan] using
    bounded_reduction a p 0 BK hBK .reverseDot (fun k => x k i.1.val) y covered

#axiomsClean bounded_reduction
#axiomsClean output_reduction
#axiomsClean query_reduction
#axiomsClean key_reduction
#axiomsClean value_reduction
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged
