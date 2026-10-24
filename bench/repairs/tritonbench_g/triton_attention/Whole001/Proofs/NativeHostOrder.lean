import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.NativeNumerical
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
open Memory
noncomputable section

private theorem scatter_point_congr (b : Buffer) (addr : ι → Nat) (values : ι → α)
    (xs : List ι) (s t : Memory α) (r : Buffer) (o : Nat) (h : s r o = t r o) :
    scatter b addr values xs s r o = scatter b addr values xs t r o := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i xs ih =>
    apply ih
    simp only [store]
    split_ifs <;> simp_all

private theorem writeTensor_point_congr (c : Shape) (s t : Memory α) (b : Buffer)
    (values : Coord c → α) (r : Buffer) (o : Nat) (h : s r o = t r o) :
    writeTensor c s b values r o = writeTensor c t b values r o :=
  scatter_point_congr _ _ _ _ _ _ _ _ h
private theorem writeRows_point_congr (c : Shape) (s t : Memory α) (b : Buffer)
    (values : Fin c.groups × Fin c.rows → α) (r : Buffer) (o : Nat) (h : s r o = t r o) :
    writeRows c s b values r o = writeRows c t b values r o :=
  scatter_point_congr _ _ _ _ _ _ _ _ h

theorem preprocess_zeroDQ (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) :
    preprocess c p owners (zeroDQ c p s) = zeroDQ c p (preprocess c p owners s) := by
  have hdata : preprocessReference c p owners (load c (zeroDQ c p s) .g)
      (saved c (zeroDQ c p s)) = preprocessReference c p owners (load c s .g) (saved c s) := by
    simp [zeroDQ,saved]
  funext r o
  by_cases hr : r = .dq
  · subst r
    simp only [preprocess,hdata]
    rw [writeRows_other _ _ _ _ _ _ (by decide : Buffer.dq ≠ .delta)]
    rw [writeTensor_other _ _ _ _ _ _ (by decide : Buffer.dq ≠ .newDO)]
    apply writeTensor_point_congr
    simp [writeRows_other,writeTensor_other]
  · rw [show zeroDQ c p (preprocess c p owners s) r o = preprocess c p owners s r o
      from writeTensor_other _ _ _ _ _ _ hr]
    simp only [preprocess,hdata]
    apply writeRows_point_congr
    apply writeTensor_point_congr
    exact writeTensor_other _ _ _ _ _ _ hr

/-- The actual backward kernel consumes an already zeroed DQ allocation. -/
def backwardFrom (c : Shape) (p : Ops α) (scale : α) (s : Memory α) : Nat → Memory α
  | 0 => s
  | n + 1 => backwardStep c p scale n (backwardFrom c p scale s n)

theorem backwardPrefix_eq_from (c : Shape) (p : Ops α) (scale : α) (s : Memory α) (n : Nat) :
    backwardPrefix c p scale s n = backwardFrom c p scale (zeroDQ c p s) n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [backwardPrefix,backwardFrom,ih]

/-- Exact Python host order: forward; zeros_like(DQ); preprocess; backward.
The equality is proved from disjoint memory effects, not assumed. -/
def hostOrderRun (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) : Memory α :=
  backwardFrom c p scale (preprocess c p owners (zeroDQ c p (forward c p owners s scale))) c.blocks

theorem hostOrderRun_eq_run (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) :
    hostOrderRun c p owners s scale = run c p owners s scale := by
  rw [hostOrderRun,preprocess_zeroDQ,run,backwardPrefix_eq_from]

#axiomsClean preprocess_zeroDQ
#axiomsClean hostOrderRun_eq_run
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
