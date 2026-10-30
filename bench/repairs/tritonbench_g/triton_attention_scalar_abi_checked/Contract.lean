import bench.repairs.tritonbench_g.triton_attention.Contract

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI
namespace Checked
open VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked

/-- The original positive scalar check remains in `scaleChecks`. The added
branch excludes only positive Python integers beyond Triton's unsigned64 ABI.
The original raw scalar object and its source precision remain unchanged. -/
def scalarBindChecks (s : ScalarABI) : Bool :=
  match s.raw with
  | .integer n => decide (n < (2^64 : Int))
  | .float64 _ => true

/-- The original Python positivity/finiteness guard on the raw object, before
the separately observed FP32 positivity/finiteness check. -/
def rawScaleChecks (s : ScalarABI) : Bool :=
  match s.raw with
  | .integer n => decide (0 < n ∧ n < (2^64 : Int))
  | .float64 bits => decide (0 < bits.toNat ∧ bits.toNat < 0x7ff0000000000000)

def scalarChecks (s : ScalarABI) : Bool := scaleChecks s && rawScaleChecks s

inductive ScalarClass where
  | constexprOne | i32 | i64 | u64 | fp32
  deriving DecidableEq

/-- Class selection for the ordinary positive raw scalar domain. This records
the JIT argument ABI, before the source multiplies its FP32 score by the scalar. -/
def scalarClass : RawScalar → ScalarClass
  | .integer n => if n = 1 then .constexprOne else if n < 2^31 then .i32 else if n < 2^63 then .i64 else .u64
  | .float64 _ => .fp32

def inputChecks (g : Geometry) (q k v : Tensor) (scale : ScalarABI) : Bool :=
  VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked.inputChecks g q k v scale &&
    rawScaleChecks scale

theorem original_checks (g : Geometry) (q k v : Tensor) (scale : ScalarABI)
    (h : inputChecks g q k v scale = true) :
    VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked.inputChecks g q k v scale = true := by
  simp only [inputChecks,Bool.and_eq_true] at h
  exact h.1

theorem integer_bound (s : ScalarABI) (n : Int) (raw : s.raw = .integer n)
    (h : scalarBindChecks s = true) : n < 2^64 := by
  simpa [scalarBindChecks,raw] using h

theorem positive_integer_abi (s : ScalarABI) (n : Int) (raw : s.raw = .integer n)
    (h : rawScaleChecks s = true) :
    0 < n ∧ ((scalarClass s.raw = .constexprOne ∧ n = 1) ∨
    (scalarClass s.raw = .i32 ∧ n ≠ 1 ∧ n < 2^31) ∨
    (scalarClass s.raw = .i64 ∧ 2^31 ≤ n ∧ n < 2^63) ∨
    (scalarClass s.raw = .u64 ∧ 2^63 ≤ n ∧ n < 2^64)) := by
  have bound : 0 < n ∧ n < 2^64 := by simpa [rawScaleChecks,raw] using h
  refine ⟨bound.1,?_⟩
  rw [raw]
  by_cases h1 : n = 1
  · exact Or.inl ⟨by simp only [scalarClass,if_pos h1],h1⟩
  right
  by_cases h32 : n < 2^31
  · exact Or.inl ⟨by simp only [scalarClass,if_neg h1,if_pos h32],h1,h32⟩
  by_cases h64 : n < 2^63
  · exact Or.inr (Or.inl ⟨by simp only [scalarClass,if_neg h1,if_neg h32,if_pos h64],by omega,h64⟩)
  · exact Or.inr (Or.inr ⟨by simp only [scalarClass,if_neg h1,if_neg h32,if_neg h64],by omega,bound.2⟩)

#axiomsClean original_checks
#axiomsClean integer_bound
#axiomsClean positive_integer_abi
end Checked
end VeriTile.Bench.Repairs.TritonAttentionScalarABI
