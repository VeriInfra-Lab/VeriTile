import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NarrowUnsignedMemoryBody
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("P",.i64),("Q",.i64)]

def source (P Q : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(P).to(tl.uint8)
    tl.store(Q, (x + x * 16).to(tl.int64))
    tl.store(Q + 1, (x + x * 16).to(tl.int64), mask=x == x)
    tl.store(Q + 2, (x + x * 16).to(tl.int64), x == x)
    value = x + x * 16
  }

/-- A direct load/cast establishes byte arithmetic, and intervening stores
cannot erase that dtype. All three store spellings preserve wrapping before
widening; negative signed inputs and aliased pointers are allowed. -/
theorem stores_preserve_byte_arithmetic {α : Type} [Inhabited α] (M : Algebra α)
    (P Q : RegionName) (s : State α) :
    ∃ t, exec M (source P Q) s = some t ∧
      t.regs .nat [] "value" = some (fun _ =>
        (((s.mem P 0).read .int % 256).toNat * 17) % 256) ∧
      (∀ i, i < 3 → (t.mem Q i).read .int =
        (((((s.mem P 0).read .int % 256).toNat * 17) % 256 : Nat) : Int)) ∧
      (∀ reg i, (reg ≠ Q ∨ 3 ≤ i) → t.mem reg i = s.mem reg i) := by
  simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,State.setReg,Option.bind,numeric,numericEq,bop,Region.cast,integralMod,
    Broadcast.leftIndex,Broadcast.rightIndex,store,State.write]
  refine ⟨?_,?_,?_⟩
  · funext i
    simp only [bop,IntegerExpr.sameShape,Broadcast.leftIndex,Broadcast.rightIndex]
    exact (by omega : ∀ n : Nat, (n + n * 16 % 256) % 256 = n * 17 % 256) _
  · intro i hi
    have cases : i = 0 ∨ i = 1 ∨ i = 2 := by omega
    rcases cases with rfl | rfl | rfl <;> simp
    all_goals exact (by omega : ∀ n : Int, (n + n * 16) % 256 = n * 17 % 256) _
  · intro reg i h
    rcases h with h | h
    · simp [h]
    · have h0 : i ≠ 0 := by omega
      have h1 : i ≠ 1 := by omega
      have h2 : i ≠ 2 := by omega
      simp [h0,h1,h2]

#axiomsClean stores_preserve_byte_arithmetic
end VeriTile.Tests.NarrowUnsignedMemoryBody
