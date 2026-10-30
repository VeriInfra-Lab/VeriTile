import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
open FP.Structural
noncomputable section

abbrev Word32 := BitVec 32

/-- One PTX add.u32 on its three 32-bit register ports. This explicit ISA model
is separate from any native hardware correspondence and from the kernel target. -/
def ptxAdd32 (left right : Word32) : Word32 := left + right

/-- Physical INT32 ports retain bit patterns; decoding the output is signed. -/
def signedAdd32 (left right : Int) : Int :=
  (ptxAdd32 (BitVec.ofInt 32 left) (BitVec.ofInt 32 right)).toInt

theorem signedAdd32_low32 (left right : Int) :
    signedAdd32 left right = (BitVec.ofInt 32 (left + right)).toInt := by
  simp only [signedAdd32, ptxAdd32, ← BitVec.ofInt_add]

theorem signedAdd32_representable (left right : Int)
    (lo : -(2^31) ≤ left + right) (hi : left + right < 2^31) :
    signedAdd32 left right = left + right := by
  rw [signedAdd32_low32]
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num at * <;> omega

/-- Primitive native conformance, if required by a native theorem, concerns
exactly one two-input word operation. It does not assume a kernel run, a dense
answer, a compiler transformation, or any matrix shape. -/
def Add32Leaf (native : Word32 → Word32 → Word32) : Prop :=
  ∀ left right, native left right = ptxAdd32 left right

theorem nativeAdd32_low32 (native : Word32 → Word32 → Word32)
    (leaf : Add32Leaf native) (left right : Int) :
    (native (BitVec.ofInt 32 left) (BitVec.ofInt 32 right)).toInt =
      (BitVec.ofInt 32 (left + right)).toInt := by
  rw [leaf]
  exact signedAdd32_low32 left right

/-- This is the existing-core expression emitted by the local quotation:
add integer carriers, retain the low unsigned word, then interpret it signed.
The explicit casts are essential; generic Int addition alone is not the model. -/
def wordAddExpr (left right : Op .int shape) : Op .int shape :=
  .intToSigned 32 (.castNatToInt (.intToUnsigned 32
    (.add .int (IntegerExpr.sameShape shape) left right)))

theorem wordAddExpr_eval {α : Type} [Inhabited α]
    (model : Algebra α) (precision : Option ComputeDType)
    (left right : Op .int shape) (state : State α)
    (leftValues rightValues : Values α .int shape)
    (hl : evalOp model precision left state = some leftValues)
    (hr : evalOp model precision right state = some rightValues) :
    evalOp model precision (wordAddExpr left right) state =
      some (fun i => signedAdd32 (leftValues i) (rightValues i)) := by
  rw [wordAddExpr, FP.Structural.eval_intToSigned]
  simp only [FP.Structural.evalOp, hl, hr, bind, pure,
    Option.bind_some, numeric, bop, IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  funext i
  rw [signedAdd32_low32]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofInt]
  norm_num
  have hn : 0 ≤ (leftValues i + rightValues i) % (4294967296 : Int) :=
    Int.emod_nonneg _ (by decide)
  simp only [integralMod, max_eq_left hn, Int.emod_emod]

#axiomsClean signedAdd32_low32
#axiomsClean signedAdd32_representable
#axiomsClean nativeAdd32_low32
#axiomsClean wordAddExpr_eval
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
