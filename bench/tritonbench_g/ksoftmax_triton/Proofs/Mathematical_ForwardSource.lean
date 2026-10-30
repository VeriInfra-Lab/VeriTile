/-
Source-audit target: _softmax
Source-audit Lean: forwardSource%
Source-audit args: inputType=tl.float32, maskType=tl.float32, destinationType=tl.float32, version=original

The dedicated source-profile checker restores the three exact pure Option
String condition bindings below before comparing the complete original AST.
This separate source check does not assert all-profile typed execution or
native admission; Kernels.lean retains its selected backward audit view.
-/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
open Lean VeriTile.Triton

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Bool → Option String → Bool → Bool → ComputeKernel

syntax "forwardSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(forwardSource%[$inputType:tritonDType, $maskType:tritonDType,
        $destinationType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let inputAddress : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| x_ptrs = X + m * $(stride_xm) + n * $(stride_xn) + k)
      else
        `(tritonStmt| x_ptrs = X + $(stride_xm) * m + n * $(stride_xn) + k)
    `(fun (Y X M : RegionName)
        (stride_ym stride_yn stride_xm stride_xn stride_m K DEPTH : Nat)
        (LOG : Bool) (MASK_TYPE : Option String) (CAUSAL IS_FP16 : Bool) =>
      let HAS_MASK : Bool := MASK_TYPE.isSome
      let MASK_QK : Bool := MASK_TYPE == some "qk"
      let MASK_BK : Bool := MASK_TYPE == some "bk"
      ComputeKernel.mk [X, M] [Y] (triton {
  tl.region X = $inputType, M = $maskType, Y = $destinationType
  m = tl.program_id(0)
  n = tl.program_id(1)
  k = tl.arange(0, $(DEPTH))
  $inputAddress
  io_mask = k < $(K)
  if CAUSAL {
    io_mask = io_mask & (k <= n)
  }
  x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
  if CAUSAL {
    off = float("-inf")
    off = (off).to(x.dtype)
    x = tl.where(k > n, off, x)
  }
  if HAS_MASK {
    if MASK_QK {
      mask_ptrs = M + n * $(stride_m) + k
    } elif MASK_BK {
      mask_ptrs = M + m * $(stride_m) + k
    }
    add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
    x += add_mask
  }
  z = x - tl.max(x, axis=0)
  if IS_FP16 {
    z = (z).to(tl.float32)
  }
  num = tl.exp(z)
  denom = tl.sum(num, axis=0)
  if LOG {
    y = z - tl.log(denom)
  } else {
    y = num / denom
  }
  y_ptrs = Y + m * $(stride_ym) + n * $(stride_yn) + k
  tl.store(y_ptrs, (y).to($destinationType:tritonDType), mask=k < $(K))
}).surfaceBody)

/-- First complete source probe, keeping every flag and original elif. -/
def fp32Original : Source := forwardSource%[tl.float32, tl.float32, tl.float32, original]

/-- Mixed mask addition changes x's carrier only under HAS_MASK. The complete
following max/subtract/IS_FP16/exp/log branches must retain that provenance. -/
def mixedOriginal : Source := forwardSource%[tl.float16, tl.float32, tl.float16, original]

/-- Exercise precision continuation from fp64 through the conditional fp32 cast. -/
def wideOriginal : Source := forwardSource%[tl.float64, tl.float16, tl.float64, original]

/-- Exercise the typed maximum producer before the conditional cast. -/
def narrowOriginal : Source := forwardSource%[tl.float16, tl.float16, tl.float16, original]

def fp32Candidate : Source := forwardSource%[tl.float32, tl.float32, tl.float32, indexCommuted]

theorem fp32_projects (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log causal fp16 : Bool) (mode : Option String) :
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).toAlgorithm? =
      .ok (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).toAlgKernel := by
  rfl

theorem io_signature (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log causal fp16 : Bool) (mode : Option String) :
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).inputs = [X, M] ∧
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).outputs = [Y] := by
  exact ⟨rfl, rfl⟩

#axiomsClean fp32_projects
#axiomsClean io_signature
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
