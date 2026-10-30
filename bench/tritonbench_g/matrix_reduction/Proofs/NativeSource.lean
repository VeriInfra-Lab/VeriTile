import bench.tritonbench_g.matrix_reduction.Proofs.SourceAddress
import bench.tritonbench_g.matrix_reduction.Proofs.Host

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton TypedSource MachineMax ConcreteCorrect
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Arbitrary binary reduction trees, including any ordering of source leaves.
The runtime's complete tree has every column; the proof permits duplicates.
No leftmost-index or signed-zero tie rule is imposed. -/
inductive Tree (N : Nat) where
  | leaf (column : Fin N)
  | node (left right : Tree N)
def Tree.leaves : Tree N → List (Fin N)
  | .leaf i => [i] | .node a b => a.leaves ++ b.leaves
def Tree.eval (max32 : Word → Word → Word) (xs : Fin N → Word) : Tree N → Word
  | .leaf i => xs i | .node a b => max32 (a.eval max32 xs) (b.eval max32 xs)
def Tree.Complete (t : Tree N) : Prop := ∀ i, i ∈ t.leaves

/-- Exact source casts and maxnum are three local scalar numerical ports.
None can select a source address, schedule a reduction or return a matrix. -/
structure Numerics where
  widen16 : BitVec 16 → Word
  narrow16 : Word → BitVec 16
  max32 : Word → Word → Word
def Numerics.promote (ops : Numerics) : (d : Storage) → Bits d → Word
  | .fp16, x => ops.widen16 x | .fp32, x => x
def Numerics.lower (ops : Numerics) : (d : Storage) → Word → Bits d
  | .fp16, x => ops.narrow16 x | .fp32, x => x

/-- Optional historical scalar modeling contracts. These are not numerical
experiment obligations of the public kernel acceptance. maxnum selects a
non-NaN operand with maximal rank on the exact promoted image. Either signed
zero may win. Half casts are specified as the existing proved bit conversion,
not as an unproved whole-kernel answer. -/
structure NumericalContracts (ops : Numerics) : Prop where
  widen : ∀ x, isNaN .fp16 x = Bool.false → ops.widen16 x = BinaryConversion.widen x
  narrow_image : ∀ x, isNaN .fp16 x = Bool.false → ops.narrow16 (BinaryConversion.widen x) = x
  max_select : ∀ d (a b : Bits d), isNaN d a = Bool.false → isNaN d b = Bool.false →
    (ops.max32 (ops.promote d a) (ops.promote d b) = ops.promote d a ∧ rank d b ≤ rank d a) ∨
    (ops.max32 (ops.promote d a) (ops.promote d b) = ops.promote d b ∧ rank d a ≤ rank d b)

theorem native_lower_promote (ops : Numerics) (laws : NumericalContracts ops)
    (d : Storage) (x : Bits d) (hn : isNaN d x = Bool.false) :
    ops.lower d (ops.promote d x) = x := by
  cases d
  · simpa [Numerics.lower,Numerics.promote,laws.widen x hn] using laws.narrow_image x hn
  · rfl

theorem tree_selected_maximum (ops : Numerics) (laws : NumericalContracts ops)
    (d : Storage) (xs : Fin N → Bits d) (hn : ∀ i, isNaN d (xs i) = Bool.false)
    (t : Tree N) : ∃ j, j ∈ t.leaves ∧
      t.eval ops.max32 (fun i => ops.promote d (xs i)) = ops.promote d (xs j) ∧
      ∀ i ∈ t.leaves, rank d (xs i) ≤ rank d (xs j) := by
  induction t with
  | leaf j => exact ⟨j,by simp [Tree.leaves],rfl,by simp [Tree.leaves]⟩
  | node left right hl hr =>
    obtain ⟨a,ha,ea,ba⟩ := hl
    obtain ⟨b,hb,eb,bb⟩ := hr
    rcases laws.max_select d (xs a) (xs b) (hn a) (hn b) with ⟨he,hba⟩ | ⟨he,hab⟩
    · refine ⟨a,by simp [Tree.leaves,ha],?_,?_⟩
      · simp only [Tree.eval,ea,eb,he]
      · intro i hi
        rcases List.mem_append.mp hi with hi | hi
        · exact ba i hi
        · exact (bb i hi).trans hba
    · refine ⟨b,by simp [Tree.leaves,hb],?_,?_⟩
      · simp only [Tree.eval,ea,eb,he]
      · intro i hi
        rcases List.mem_append.mp hi with hi | hi
        · exact (ba i hi).trans hab
        · exact bb i hi

def rowResult (ops : Numerics) (d : Storage) (xs : Fin N → Bits d) (t : Tree N) : Bits d :=
  ops.lower d (t.eval ops.max32 (fun i => ops.promote d (xs i)))

theorem complete_tree_maximum (ops : Numerics) (laws : NumericalContracts ops)
    (d : Storage) (xs : Fin N → Bits d) (hn : ∀ i, isNaN d (xs i) = Bool.false)
    (t : Tree N) (complete : t.Complete) :
    IsMaximum d xs (rowResult ops d xs t) ∧ isNaN d (rowResult ops d xs t) = Bool.false := by
  obtain ⟨j,_,he,hb⟩ := tree_selected_maximum ops laws d xs hn t
  have hout : rowResult ops d xs t = xs j := by
    rw [rowResult,he,native_lower_promote ops laws d (xs j) (hn j)]
  rw [hout]
  exact ⟨⟨⟨j,rfl⟩,fun i => hb i (complete i)⟩,hn j⟩

/-- The public value relation remains true for every reduction tree and either
zero sign; exact equality with the software leftmost tie choice is unnecessary. -/
theorem native_source_rank (ops : Numerics) (laws : NumericalContracts ops)
    (d : Storage) (xs : Fin N → Bits d) (hn : ∀ i, isNaN d (xs i) = Bool.false)
    (t : Tree N) (complete : t.Complete) (hN : 0 < N) :
    rank d (rowResult ops d xs t) = rank d (rowMaximum d xs) := by
  apply Host.maximum_rank_eq d xs
  · exact (complete_tree_maximum ops laws d xs hn t complete).1
  · simpa [rowMaximum,hN] using best_spec d xs hN

#axiomsClean input_address_exact
#axiomsClean output_address_exact
#axiomsClean tree_selected_maximum
#axiomsClean complete_tree_maximum
#axiomsClean native_source_rank
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native
