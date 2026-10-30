import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Native

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Cells
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section

/-- Independent cast-erasure memory. Cells contain arbitrary mathematical
Reals, not codes of a floating format. There is no encoding into finite bytes. -/
abbrev Heap := Nat → ℝ

def primitives (iw ow : Nat) (hi : 0 < iw) (ho : 0 < ow) :
    Memory.Primitives Heap ℝ ℝ iw ow where
  readInput := fun m q => m q
  readOutput := fun m q => m q
  writeOutput := fun m q v => Function.update m q v
  byte := fun _ _ => 0
  read_same := by intros; simp
  read_disjoint := by
    intro m a v b apart
    have ne : b≠a := by
      rcases apart with h|h <;> omega
    simp [ne]
  input_disjoint := by
    intro m a v b apart
    have ne : b≠a := by
      rcases apart with h|h <;> omega
    simp [ne]
  byte_frame := by intros; rfl

/-- The full Real-cell frame is proved from concrete updates separately. A
constant finite byte observation could not establish this statement. -/
theorem stores_frame (iw ow : Nat) (hi : 0 < iw) (ho : 0 < ow)
    (base : Nat) (values : Nat → ℝ) (jobs : List Nat) (s : Heap) (q : Nat)
    (outside : ∀ j∈jobs,q≠Memory.address base ow j) :
    Memory.stores (primitives iw ow hi ho) base values jobs s q=s q := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [Memory.stores,ih _ (fun k hk => outside k (by simp [hk]))]
    exact Function.update_of_ne (outside j (by simp)) _ _

/-- Identity is the independent mathematical target after erasing conversion
precision. The input value is quantified over every Real at every coordinate. -/
theorem complete (twice : Bool) (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (apart : Native.Separate x y) (before : Heap) :
    (∀ i,i<n → Memory.run (primitives iw ow x.positive y.positive) twice id
      x.base y.base (Chunking.batches n) before (Memory.address y.base ow i)=
      before (Memory.address x.base iw i)) ∧
    (∀ q,¬y.Active q → Memory.run (primitives iw ow x.positive y.positive) twice id
      x.base y.base (Chunking.batches n) before q=before q) := by
  constructor
  · exact (Native.full_execution (primitives iw ow x.positive y.positive)
      twice id x y apart before).1
  · intro q outside
    rw [Memory.buffered (primitives iw ow x.positive y.positive) twice id x.base y.base n (Chunking.batches n) before before
      (fun i hi => (Chunking.full_coverage n i).mp hi)
      (fun i hi j hj => Native.separate_accesses x y apart i j hi hj) (fun _ _ => rfl)]
    apply stores_frame
    intro i hi same
    have active : i<n := (Chunking.full_coverage n i).mp
      ((Memory.mem_schedule twice (Chunking.batches n) i).mp hi)
    have product := Nat.mul_lt_mul_of_pos_right active y.positive
    apply outside
    unfold Native.Allocation.Active
    rw [same]
    unfold Memory.address
    omega

#axiomsClean complete
#axiomsClean stores_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Cells
