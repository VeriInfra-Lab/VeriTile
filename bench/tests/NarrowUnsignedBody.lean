import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NarrowUnsignedBody
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  constexpr := ["BLOCK"]

def packed (BLOCK : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lane = tl.arange(0, BLOCK)
    byte = lane.to(tl.uint8)
    copied = byte
    packed = byte + copied * 16
    widened = packed.to(tl.int64)
    after = widened + 1000
  }

/-- Byte arithmetic wraps at the original width before the result is widened.
Every lane and arbitrary initial memory are covered, including overflow. -/
theorem byte_arithmetic_before_widening {α : Type} [Inhabited α] (M : Algebra α)
    (BLOCK : Nat) (s : State α) :
    ∃ t, exec M (packed BLOCK) s = some t ∧
      t.regs .nat [BLOCK] "packed" = some (fun i =>
        ((BitVec.ofNat 8 i.1.val) + (BitVec.ofNat 8 i.1.val) * 16).toNat) ∧
      t.regs .int [BLOCK] "after" = some (fun i =>
        (((BitVec.ofNat 8 i.1.val) + (BitVec.ofNat 8 i.1.val) * 16).toNat : Int) + 1000) ∧
      t.mem = s.mem := by
  simp [packed,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,bop,Option.bind,State.setReg,integralMod,
    Broadcast.leftIndex,Broadcast.rightIndex,BitVec.toNat_add,BitVec.toNat_mul,
    BitVec.toNat_ofNat,Nat.add_mod,Nat.mul_mod]
  constructor <;> funext i <;> simp [bop,IntegerExpr.sameShape,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mod,Nat.mul_mod,Nat.mod_mod]

#axiomsClean byte_arithmetic_before_widening
end VeriTile.Tests.NarrowUnsignedBody
