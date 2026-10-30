import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FPDotRebinding
open VeriTile.Triton

def dotSource (A B : RegionName) : ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32
  i = tl.arange(0, 2)
  a = tl.load(A + i[:, None] * 2 + i[None, :])
  b = tl.load(B + i[:, None] * 2 + i[None, :])
  score = tl.dot(a, b, allow_tf32=false)
  square = score * score
}

def isFP32Assignment : Option ComputeStmt → Bool
  | some (.assign _ _ _ (.compute (.alg .fp32 _))) => Bool.true
  | _ => Bool.false

/-- No later fp32 operand conceals lost precision on either the dot or its
result: both use the explicitly supplied source float32 type. -/
theorem dot_and_result_keep_precision (A B : RegionName) :
    isFP32Assignment ((dotSource A B).surfaceBody[3]?) = Bool.true ∧
    isFP32Assignment ((dotSource A B).surfaceBody[4]?) = Bool.true := ⟨rfl,rfl⟩

def reboundSource (dtype : FloatDType) (A B Z S : RegionName) (count : Nat) : ComputeKernel := triton {
  tl.region A = DTYPE, B = tl.int32, Z = tl.int32, S = DTYPE
  a = tl.load(A)
  b = tl.load(B)
  zeros = tl.load(Z)
  scales = tl.load(S)
  nibble = (b >> $(count)) & 15
  zeros = (zeros >> $(count)) & 15
  zeros = tl.toReal(zeros).to(DTYPE) * scales
  b = (tl.toReal(nibble).to(DTYPE) * scales).to(a.dtype)
}

def isFloatAssignment (dtype : TileDType) : Option ComputeStmt → Bool
  | some (.assign actual _ _ _) => actual == dtype
  | _ => Bool.false

/-- Reusing a signed packed-word name for an explicitly converted half value
must not force the floating expression back through Nat address inference. -/
theorem rebound_registers_keep_float16 (A B Z S : RegionName) (count : Nat) :
    isFloatAssignment .fp16 ((reboundSource .fp16 A B Z S count).surfaceBody[6]?) = Bool.true ∧
    isFloatAssignment .fp16 ((reboundSource .fp16 A B Z S count).surfaceBody[7]?) = Bool.true := ⟨rfl,rfl⟩

#axiomsClean dot_and_result_keep_precision
#axiomsClean rebound_registers_keep_float16
end VeriTile.Tests.FPDotRebinding
