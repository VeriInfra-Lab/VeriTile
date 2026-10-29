import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.Erasure

/-! Operational direct-call checks. These do not claim source frontend or
alias-safe tuple lowering, which require separate source regressions. -/
namespace VeriTile.Bench.Tests.DirectPhiloxOp
open VeriTile.Triton
open Random.Direct

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

private theorem real_philox {ins outs : Shapes} {n : Nat} (cfg : Profile)
    (plan : Plan ins n outs) (field : Field) (seed : Op .int [])
    (c0 : Op .int ins.c0) (c1 : Op .int ins.c1)
    (c2 : Op .int ins.c2) (c3 : Op .int ins.c3) (s : BlockState) :
    evalOp (.philox4x32 cfg plan field seed c0 c1 c2 c3) s = (do
      let vs ← evalOp seed s
      let v0 ← evalOp c0 s
      let v1 ← evalOp c1 s
      let v2 ← evalOp c2 s
      let v3 ← evalOp c3 s
      let result ← evalField cfg plan field (vs.data PUnit.unit)
        ⟨v0.data, v1.data, v2.data, v3.data⟩
      pure (⟨result⟩ : Tile .nat (outs.at field))) := by
  simp only [evalOp]


private theorem rounded_philox {ins outs : Shapes} {n : Nat} (R : RoundingModel) (cfg : Profile)
    (plan : Plan ins n outs) (field : Field) (seed : Op .int [])
    (c0 : Op .int ins.c0) (c1 : Op .int ins.c1)
    (c2 : Op .int ins.c2) (c3 : Op .int ins.c3) (s : BlockState) :
    evalOpR R (.philox4x32 cfg plan field seed c0 c1 c2 c3) s = (do
      let vs ← evalOpR R seed s
      let v0 ← evalOpR R c0 s
      let v1 ← evalOpR R c1 s
      let v2 ← evalOpR R c2 s
      let v3 ← evalOpR R c3 s
      let result ← evalField cfg plan field (vs.data PUnit.unit)
        ⟨v0.data, v1.data, v2.data, v3.data⟩
      pure (⟨result⟩ : Tile .nat (outs.at field))) := by
  simp only [evalOpR]


def unsignedProfile : Profile := ⟨.i64, .u32, .u32, .u32, .u32⟩
def signedProfile : Profile := ⟨.i64, .i32, .i32, .i32, .i32⟩
def scalarShapes : Shapes := ⟨[], [], [], []⟩
def emptyShapes : Shapes := ⟨[0], [], [0], []⟩
def emptyOutputs : Shapes := ⟨[0], [0], [0], [0]⟩

def emptyPlan : Plan emptyShapes 1 emptyOutputs :=
  .step .scalarR .scalarR .done

def zeroRoundSigned : Op .nat [] :=
  .philox4x32 (ins := scalarShapes) signedProfile .done .c0
    (.constInt (-1)) (.constInt (-1)) (.constInt 0) (.constInt 0) (.constInt 0)

theorem signed_word_bitcast (s : BlockState) :
    evalOp zeroRoundSigned s = some (Tile.scalar 0xffffffff) := by
  apply (real_philox signedProfile (.done : Plan scalarShapes 0 scalarShapes) .c0
    (.constInt (-1)) (.constInt (-1)) (.constInt 0) (.constInt 0) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem signed_word_bitcast_rounded (R : RoundingModel) (s : BlockState) :
    evalOpR R zeroRoundSigned s = some (Tile.scalar 0xffffffff) := by
  apply (rounded_philox R signedProfile (.done : Plan scalarShapes 0 scalarShapes) .c0
    (.constInt (-1)) (.constInt (-1)) (.constInt 0) (.constInt 0) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOpR, bind, Option.bind, Tile.scalar]
  rfl

theorem signed_word_bitcast_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p zeroRoundSigned s = some (fun _ => 0xffffffff) := by
  simp only [zeroRoundSigned, FP.Structural.evalOp, bind, Option.bind]
  rfl

def zeroRoundScalarHigh : Op .nat [] :=
  .philox4x32 (ins := emptyShapes) unsignedProfile .done .c1
    (.constInt 0) (.full [0] (.constInt 0)) (.constInt 123)
    (.full [0] (.constInt 0)) (.constInt 0)

theorem zero_round_keeps_scalar_high (s : BlockState) :
    evalOp zeroRoundScalarHigh s = some (Tile.scalar 123) := by
  apply (real_philox unsignedProfile (.done : Plan emptyShapes 0 emptyShapes) .c1
    (.constInt 0) (.full [0] (.constInt 0)) (.constInt 123)
    (.full [0] (.constInt 0)) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem zero_round_keeps_scalar_high_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p zeroRoundScalarHigh s = some (fun _ => 123) := by
  simp only [zeroRoundScalarHigh, FP.Structural.evalOp, bind, Option.bind]
  rfl

def emptyOutput (seed c1 c3 : Op .int []) : Op .nat [0] :=
  .philox4x32 unsignedProfile emptyPlan .c0 seed
    (.full [0] (.constInt 0)) c1 (.full [0] (.constInt 0)) c3

theorem invalid_seed_before_empty_broadcast (s : BlockState) :
    evalOp (emptyOutput (.constInt (2^63)) (.constInt 0) (.constInt 0)) s = none := by
  apply (real_philox unsignedProfile emptyPlan .c0 (.constInt (2^63))
    (.full [0] (.constInt 0)) (.constInt 0) (.full [0] (.constInt 0)) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem invalid_scalar_counter_before_empty_broadcast (s : BlockState) :
    evalOp (emptyOutput (.constInt 0) (.constInt (2^32)) (.constInt 0)) s = none := by
  apply (real_philox unsignedProfile emptyPlan .c0 (.constInt 0)
    (.full [0] (.constInt 0)) (.constInt (2^32)) (.full [0] (.constInt 0)) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem invalid_last_counter_before_empty_broadcast (s : BlockState) :
    evalOp (emptyOutput (.constInt 0) (.constInt 0) (.constInt (-1))) s = none := by
  apply (real_philox unsignedProfile emptyPlan .c0 (.constInt 0)
    (.full [0] (.constInt 0)) (.constInt 0) (.full [0] (.constInt 0)) (.constInt (-1)) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem invalid_scalar_counter_rounded (R : RoundingModel) (s : BlockState) :
    evalOpR R (emptyOutput (.constInt 0) (.constInt (2^32)) (.constInt 0)) s = none := by
  apply (rounded_philox R unsignedProfile emptyPlan .c0 (.constInt 0)
    (.full [0] (.constInt 0)) (.constInt (2^32)) (.full [0] (.constInt 0)) (.constInt 0) s).trans
  dsimp only [scalarShapes, emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOpR, bind, Option.bind, Tile.scalar]
  rfl

theorem invalid_scalar_counter_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p
      (emptyOutput (.constInt 0) (.constInt (2^32)) (.constInt 0)) s = none := by
  simp only [emptyOutput, FP.Structural.evalOp, bind, Option.bind]
  rfl

theorem missing_last_operand_is_evaluated (s : BlockState)
    (h : s.regs .int [] "missing" = none) :
    evalOp (emptyOutput (.constInt 0) (.constInt 0) (.ref .int [] "missing")) s = none := by
  apply (real_philox unsignedProfile emptyPlan .c0 (.constInt 0)
    (.full [0] (.constInt 0)) (.constInt 0) (.full [0] (.constInt 0)) (.ref .int [] "missing") s).trans
  dsimp only [emptyShapes, emptyOutputs, Shapes.at]
  simp only [evalOp, h, bind, Option.bind]

theorem missing_last_operand_is_evaluated_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α)
    (h : s.regs .int [] "missing" = none) :
    FP.Structural.evalOp M p
      (emptyOutput (.constInt 0) (.constInt 0) (.ref .int [] "missing")) s = none := by
  simp only [emptyOutput, emptyShapes, emptyOutputs, Shapes.at, FP.Structural.evalOp, h, bind, Option.bind]

def halfWay : Op .real [] := .uniform32 (.constNat 0x01000001)

theorem halfway_bits :
    (Random.uniform 0x01000001).bits = (0x3bffffff : BitVec 32) := by decide +kernel

theorem uniform_retains_both_roundings {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p halfWay s = some (fun _ => M.fp32Bits ⟨0x3bffffff⟩) := by
  change some (fun _ => M.fp32Bits ⟨(Random.uniform 0x01000001).bits⟩) = _
  rw [halfway_bits]

theorem uniform_actual_real_projection (s : BlockState) :
    evalOp halfWay s = some (Tile.scalar (some (Random.uniformReal 0x01000001))) := by
  simp only [halfWay, evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem uniform_invalid_word_rejected (s : BlockState) :
    evalOp (.uniform32 (.constNat (2^32))) s = none := by
  simp only [evalOp, bind, Option.bind, Tile.scalar]
  rfl

theorem uniform_invalid_word_rejected_typed {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (.uniform32 (.constNat (2^32))) s = none := by
  simp only [FP.Structural.evalOp, bind, Option.bind]
  rfl

theorem uniform_missing_empty_child_rejected (s : BlockState)
    (h : s.regs .nat [0] "missing" = none) :
    evalOp (.uniform32 (.ref .nat [0] "missing")) s = none := by
  simp only [evalOp, h, bind, Option.bind]

theorem uniform_projection_finite (word : Random.Word) :
    (Random.uniform word).toRat? = some (Random.uniformRat word) :=
  Random.uniform_toRat_uniformRat word

#print axioms signed_word_bitcast
#print axioms signed_word_bitcast_rounded
#print axioms signed_word_bitcast_typed
#print axioms zero_round_keeps_scalar_high
#print axioms zero_round_keeps_scalar_high_typed
#print axioms invalid_seed_before_empty_broadcast
#print axioms invalid_scalar_counter_before_empty_broadcast
#print axioms invalid_last_counter_before_empty_broadcast
#print axioms invalid_scalar_counter_rounded
#print axioms invalid_scalar_counter_typed
#print axioms missing_last_operand_is_evaluated
#print axioms missing_last_operand_is_evaluated_typed
#print axioms halfway_bits
#print axioms uniform_retains_both_roundings
#print axioms uniform_actual_real_projection
#print axioms uniform_invalid_word_rejected
#print axioms uniform_invalid_word_rejected_typed
#print axioms uniform_missing_empty_child_rejected
#print axioms uniform_projection_finite
end VeriTile.Bench.Tests.DirectPhiloxOp
