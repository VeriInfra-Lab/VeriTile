import bench.tritonbench_g.bmm_optimized.Sources.Spec

namespace VeriTile.Bench.Repairs.BmmOptimized
open scoped BigOperators

theorem laneLoop_add {α : Type} (s : Shape) (p : Scalar α) (m : Memory α)
    (x : Cell) (start n extra : Nat) (carry : α) :
    laneLoop s p m x start (n+extra) carry =
      laneLoop s p m x (start+n) extra (laneLoop s p m x start n carry) := by
  induction extra with
  | zero => simp [laneLoop]
  | succ extra ih =>
    change p.fma (aLoad s p m x (start+(n+extra)))
        (bLoad s p m x (start+(n+extra))) (laneLoop s p m x start (n+extra) carry) =
      p.fma (aLoad s p m x ((start+n)+extra))
        (bLoad s p m x ((start+n)+extra))
        (laneLoop s p m x (start+n) extra (laneLoop s p m x start n carry))
    rw [ih,Nat.add_assoc]

theorem blockLoop_flat {α : Type} (s : Shape) (p : Scalar α) (m : Memory α)
    (x : Cell) (n : Nat) :
    blockLoop s p m x n = laneLoop s p m x 0 (n*32) p.zero := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [blockLoop,ih,show (n+1)*32=n*32+32 by omega,laneLoop_add]
    simp only [Nat.zero_add]

theorem laneLoop_reference {α : Type} (s : Shape) (p : Scalar α) (m : Memory α)
    (x : Cell) (active : s.Active x) (n : Nat) :
    laneLoop s p m x 0 n p.zero = reference s p m x n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [laneLoop,reference,Nat.zero_add,ih,aLoad,bLoad,active.2.1,active.2.2,
      true_and,and_true]

theorem source_ordered {α : Type} (s : Shape) (p : Scalar α) (m : Memory α)
    (x : Cell) (active : s.Active x) :
    sourceValue s p m x=ordered s p m x := by
  rw [sourceValue,blockLoop_flat,laneLoop_reference s p m x active]
  rfl

theorem padded_covers (k : Nat) : k≤((k+31)/32)*32 := by omega
theorem padded_bound (k : Nat) : ((k+31)/32)*32<k+32 := by omega

theorem reference_real (s : Shape) (m : Memory ℝ) (x : Cell) (n : Nat) :
    reference s realScalar m x n =
      ∑ k ∈ Finset.range n,
        if k<s.k then m .a (s.a x.batch x.row k) * m .b (s.b x.batch k x.column) else 0 := by
  induction n with
  | zero => simp [reference,realScalar]
  | succ n ih =>
    rw [reference,Finset.sum_range_succ,ih]
    by_cases hk : n<s.k <;> simp [realScalar,hk] <;> ring

theorem sum_zero_tail (f : Nat → ℝ) (k n : Nat) (h : k≤n) :
    (∑ i ∈ Finset.range n,if i<k then f i else 0)=∑ i ∈ Finset.range k,f i := by
  have eq : (∑ i ∈ Finset.range k,if i<k then f i else 0)=
      ∑ i ∈ Finset.range n,if i<k then f i else 0 := by
    apply Finset.sum_subset (Finset.range_mono h)
    intro i _ outside
    have hi : ¬i<k := by simpa only [Finset.mem_range] using outside
    simp [hi]
  rw [←eq]
  apply Finset.sum_congr rfl
  intro i hi
  simp [Finset.mem_range.mp hi]

theorem ordered_real (s : Shape) (m : Memory ℝ) (x : Cell) :
    ordered s realScalar m x=mathematical s m x := by
  rw [ordered,reference_real,sum_zero_tail _ s.k _ (padded_covers s.k)]
  rfl

theorem source_real (s : Shape) (m : Memory ℝ) (x : Cell) (active : s.Active x) :
    sourceValue s realScalar m x=mathematical s m x := by
  rw [source_ordered s realScalar m x active,ordered_real]

#axiomsClean laneLoop_add
#axiomsClean blockLoop_flat
#axiomsClean source_ordered
#axiomsClean source_real
end VeriTile.Bench.Repairs.BmmOptimized
