import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Precise source division keeps its independent primitive identity.
Source: https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html
These checks use observable callback traces, with no numerical equality or
native/compiler conformance assumption. -/
namespace VeriTile.Tests.DivRNSource
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("O",.fp32)]

def source (A B O : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    a = tl.load(A)
    b = tl.load(B)
    z = tl.div_rn(a, b)
    tl.store(O, z)
  }

theorem precise_node (A B O : RegionName) :
    (source A B O).surfaceBody[2]? = some
      (.assign .real [] "z" (.compute (.alg .fp32
        (.divRN .nil (.ref .real [] "a") (.ref .real [] "b"))))) := rfl

theorem erasure_keeps_identity (bc : Broadcast a b out) (x : Op .real a) (y : Op .real b) :
    (Op.divRN bc x y).eraseDType = .divRN bc x.eraseDType y.eraseDType := by
  simp only [Op.eraseDType]

inductive Trace where
  | left | right | ordinary (op : Binary) | precise (a b : Trace)
  deriving Inhabited, DecidableEq

def model : Algebra Trace where
  literal _ _ _ := .left
  negInf := .left
  binary _ _ op _ _ := .ordinary op
  unary _ _ _ := .left
  cast _ _ _ := id
  fromNat _ _ := .left
  fromInt _ _ := .left
  fp32Bits _ := .left
  fp32Load := id
  reduceMax _ {_} _ _ _ := fun _ => .left
  reduceSum _ {_} _ _ _ := fun _ => .left
  divRN := some Trace.precise

def empty : State Trace where
  mem _ _ := .mk .real .left
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

def entry := (empty.setReg "a" .real [] (fun _ => .left)).setReg "b" .real [] (fun _ => .right)

theorem distinct_callbacks :
    evalOp model none (.divRN .nil (.ref .real [] "a") (.ref .real [] "b")) entry =
      some (fun _ => Trace.precise .left .right) ∧
    evalOp model none (.div .real .nil (.ref .real [] "a") (.ref .real [] "b")) entry =
      some (fun _ => Trace.ordinary .div) := ⟨rfl,rfl⟩

theorem no_ordinary_fallback :
    evalOp {model with divRN := none} none
      (.divRN .nil (.ref .real [] "a") (.ref .real [] "b")) entry = none := rfl

theorem empty_tile_still_requires_primitive (M : Algebra Trace) (s : State Trace)
    (missing : M.divRN=none) :
    evalOp M none (.divRN .scalarR (.full [0] (.const 0)) (.const 1)) s=none := by
  simp [evalOp,missing]

theorem broadcast_operands (M : Algebra Trace) (s : State Trace) (n : Nat)
    (x : Op .real [n]) (y : Op .real [])
    (xs : Values Trace .real [n]) (yv : Trace) (f : Trace → Trace → Trace)
    (hx : evalOp M none x s=some xs) (hy : evalOp M none y s=some (fun _ => yv))
    (hf : M.divRN=some f) :
    evalOp M none (.divRN .scalarR x y) s=some (fun i => f (xs i) yv) := by
  simp only [evalOp,hx,hy,hf,Option.bind_some]
  rfl

/-- error: tl.div_rn requires known FP32 source operands; cast other formats explicitly -/
#guard_msgs in
example (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp64),("B",.fp64)]} : CallSignature) in triton {
    a = tl.load(A)
    b = tl.load(B)
    z = tl.div_rn(a, b)
  }

/-- error: tl.div_rn requires known FP32 source operands; cast other formats explicitly -/
#guard_msgs in
example (dtype : FloatDType) (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.inputFloat),("B",.inputFloat)]} : CallSignature) in triton {
    a = tl.load(A)
    b = tl.load(B)
    z = tl.div_rn(a, b)
  }

/-- error: tl.div_rn requires known FP32 source operands; cast other formats explicitly -/
#guard_msgs in
example (A B O : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp64),("B",.fp64),("O",.fp32)]} : CallSignature) in triton {
    a = tl.load(A)
    b = tl.load(B)
    z = (0.0).to(tl.float32)
    tl.store(O, z, mask=tl.div_rn(a, b) > 0.0)
  }

/-- Explicit source conversions can establish the supported input format. -/
def convertedHalfInputs (A B O : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp16),("B",.fp16),("O",.fp32)]} : CallSignature) in triton {
    a = tl.load(A).to(tl.float32)
    b = tl.load(B).to(tl.float32)
    z = tl.div_rn(a, b)
    tl.store(O, z)
  }

/-- The precise intrinsic must retain a nested FP64 product before its
explicit FP32 conversion; the intrinsic's own context cannot retag that product. -/
def convertedProduct (A B O : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp64),("B",.fp32),("O",.fp32)]} : CallSignature) in triton {
    a = tl.load(A)
    b = tl.load(B)
    z = tl.div_rn((a * a).to(tl.float32), b)
    tl.store(O, z)
  }

theorem nested_precision_preserved (A B O : RegionName) :
    (convertedProduct A B O).surfaceBody[2]? = some
      (.assign .real [] "z" (.compute (.alg .fp32
        (.divRN .nil
          (.castFloat .real .real (.withPrecision .real .fp64
            (.mul .real .nil (.ref .real [] "a") (.ref .real [] "a"))))
          (.ref .real [] "b"))))) := rfl

#axiomsClean precise_node
#axiomsClean erasure_keeps_identity
#axiomsClean distinct_callbacks
#axiomsClean no_ordinary_fallback
#axiomsClean empty_tile_still_requires_primitive
#axiomsClean broadcast_operands
#axiomsClean nested_precision_preserved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.DivRNSource
