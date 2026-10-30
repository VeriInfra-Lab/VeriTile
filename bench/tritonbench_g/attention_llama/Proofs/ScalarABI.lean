import bench.tritonbench_g.attention_llama.Contract

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.ScalarABI
open VeriTile Triton
noncomputable section

inductive Argument where
  | one
  | i32 (word : BitVec 32)
  | i64 (word : BitVec 64)
  | u64 (word : BitVec 64)
  | binary64 (bits : BitVec 64)
  deriving DecidableEq

/-- Actual unannotated scalar binder dispatch, including integer-one
specialization. The binary64 constructor retains Python's raw payload before
its native FP32 parameter conversion. -/
def specialize : RawScalar → Option Argument
  | .integer n =>
      if n=1 then some .one
      else if -(2^31 : Int)≤n ∧ n<2^31 then some (.i32 (BitVec.ofInt 32 n))
      else if -(2^63 : Int)≤n ∧ n<2^63 then some (.i64 (BitVec.ofInt 64 n))
      else if 2^63≤n ∧ n<2^64 then some (.u64 (BitVec.ofInt 64 n))
      else none
  | .float64 bits => some (.binary64 bits)

 theorem succeeds_iff (raw : RawScalar) :
    (∃ a, specialize raw=some a) ↔ scalarABICheck Bool.true raw=Bool.true := by
  cases raw with
  | float64 bits => simp [specialize,scalarABICheck]
  | integer n =>
    simp only [launched_integer_iff,specialize]
    split_ifs <;> simp_all <;> omega

/-- Individual ABI conversion instructions, never an attention-output port.
The integer paths convert directly from their selected signed/unsigned words;
no intermediate binary64 normalization is introduced. -/
structure Conversions (α : Type) where
  one : α
  i32 : BitVec 32 → α
  i64 : BitVec 64 → α
  u64 : BitVec 64 → α
  binary64ToFP32 : BitVec 64 → α

def convert (C : Conversions α) : Argument → α
  | .one => C.one
  | .i32 w => C.i32 w
  | .i64 w => C.i64 w
  | .u64 w => C.u64 w
  | .binary64 bits => C.binary64ToFP32 bits

/-- Mathematical observation of the original Python scalar. Finite binary64
values are decoded exactly, including subnormals and values outside FP32's
range. This observation is separate from all native conversion instructions. -/
def realValue : RawScalar → Option ℝ
  | .integer n => some (n : ℝ)
  | .float64 bits =>
      let exponent : Nat := bits.toNat / 2^52 % 2^11
      let fraction : Nat := bits.toNat % 2^52
      let sign : ℝ := if bits.toNat / 2^63=0 then 1 else -1
      if exponent=2047 then none
      else if exponent=0 then some (sign*(fraction : ℝ)*(2 : ℝ)^(-1074 : Int))
      else some (sign*((2^52+fraction : Nat) : ℝ)*(2 : ℝ)^((exponent : Int)-1075))

 theorem signed32_exact (n : Int) (h : -(2^31 : Int)≤n ∧ n<2^31) :
    (BitVec.ofInt 32 n).toInt=n := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num at * <;> omega
 theorem signed64_exact (n : Int) (h : -(2^63 : Int)≤n ∧ n<2^63) :
    (BitVec.ofInt 64 n).toInt=n := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num at * <;> omega

#axiomsClean succeeds_iff
#axiomsClean signed32_exact
#axiomsClean signed64_exact
#specNonCircular specialize avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular realValue avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.ScalarABI
