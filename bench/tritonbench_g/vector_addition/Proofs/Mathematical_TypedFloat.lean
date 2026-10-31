import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_HostGrid

/-! Typed materialization of the original elementwise load/add/store graph.
The five storage formats below have distinct existing AST channels. Operand
promotion and the conversion back to x's output dtype are explicit. This is
not fp64 or fnuz coverage, nor proof of native compiler/allocator support.
Conversions between distinct formats and floating addition are opaque
numerical primitives; identity conversions are defined as identity.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters offset)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | fp16 | bf16 | fp32 | f8e4 | f8e5
  deriving DecidableEq, Repr

def Storage.dtype : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32
  | .f8e4 => .f8e4 | .f8e5 => .f8e5

/-- Triton 3.7.1 tensor/tensor floating promotion, restricted to the five
represented storage formats. The original output is empty_like(x). -/
def promote (x y : Storage) : Storage :=
  if x = .fp32 ∨ y = .fp32 then .fp32
  else if x = .fp16 ∨ y = .fp16 then .fp16
  else if x = .bf16 ∧ y = .bf16 then .bf16
  else if x = .bf16 ∨ y = .bf16 then .fp32
  else if x = y then x else .fp16

def castBackend {α : Type} (M : Algebra α) : Algebra α :=
  { M with cast := fun p src dst a => if src = dst then a else M.cast p src dst a }

def castValue {α : Type} (M : Algebra α) (src dst : FloatDType) (a : α) : α :=
  if src = dst then a else M.cast none src dst a

@[simp] theorem toFloat_ofFloat {α : Type} (d : FloatDType) (a : α) :
    toFloat d (ofFloat d a) = a := by cases d <;> rfl

@[simp] theorem ofFloat_toFloat {α : Type} (d : FloatDType) (a : Value α d.toTileDType) :
    ofFloat d (toFloat d a) = a := by cases d <;> rfl

@[simp] theorem toFloat_numeric {α : Type} (M : Algebra α) (d : FloatDType)
    (a b : Value α d.toTileDType) :
    toFloat d (numeric M none .add d.numericDType a b) =
      M.binary none d .add (toFloat d a) (toFloat d b) := by
  cases d <;> rfl

@[simp] theorem numeric_nat_add {α : Type} (M : Algebra α) (a b : Nat) :
    numeric M none .add NumericDType.nat a b = a+b := rfl

@[simp] theorem numeric_nat_mul {α : Type} (M : Algebra α) (a b : Nat) :
    numeric M none .mul NumericDType.nat a b = a*b := rfl

def resultCell {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk dx.toTileDType (ofFloat dx (castValue M dc dx
    (M.binary none dc .add
      (castValue M dx dc (toFloat dx ((mem p.x o).read dx.toTileDType)))
      (castValue M dy dc (toFloat dy ((mem p.y o).read dy.toTileDType))))))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then
        resultCell M dx dy dc p s.mem (offset p s i) else mem r o
    else mem) s.mem

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) :
    expectedMemory M dx dy dc p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell M dx dy dc p s.mem (offset p s i)) else t) s).mem := by
  rw [ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory M dx dy dc p s p.output (offset p s i) = resultCell M dx dy dc p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M dx dy dc p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (ValueProofs.offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : ValueProofs.Outside p s r o) :
    expectedMemory M dx dy dc p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M dx dy dc p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell M dx dy dc p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M dx dy dc p s i (by rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      unfold offset at hi ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

#specNonCircular expectedMemory avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean expected_cell
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
