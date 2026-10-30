import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_CounterOperands
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TypedRandom

/-! The source's signed64 mask/shift expressions in the typed floating
evaluator. These laws hold for every floating algebra and use the same
machine-word identities as the independent scalar specification.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedCounterOperands
open VeriTile.Triton ScalarSpec FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

theorem eval_low_mask {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (value : Int) (valueOp : Op .int []) (s : State α)
    (hv : FP.Structural.evalOp M p valueOp s = some (fun _ => value)) :
    FP.Structural.evalOp M p
      (.intToUnsigned 32 (.intBitAnd 64 .nil valueOp (.constInt 0xffffffff))) s =
        some (fun _ => (BitVec.ofInt 32 value).toNat) := by
  rw [FP.Structural.eval_intToUnsigned, FP.Structural.eval_intBitAnd, hv]
  simp only [evalOp_unfold, bind, Option.bind, pure]
  apply congrArg some
  funext i
  change (((((BitVec.ofInt 64 value) &&& BitVec.ofInt 64 0xffffffff).toInt) %
    (2^32 : Int)).toNat) = _
  rw [unsigned_cast_eq_bitvec]
  change (BitVec.ofInt 32
    (((BitVec.ofInt 64 value) &&& BitVec.ofNat 64 0xffffffff).toInt)).toNat = _
  rw [CounterOperands.masked_low32]

theorem eval_high_mask {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (offset : Int) (offsetOp : Op .int []) (s : State α)
    (ho : FP.Structural.evalOp M p offsetOp s = some (fun _ => offset)) :
    FP.Structural.evalOp M p (.intToUnsigned 32 (.intBitAnd 64 .nil
      (.intShiftRight .nil offsetOp (.constNat 32)) (.constInt 0xffffffff))) s =
        some (fun _ => (counterHigh offset).toNat) := by
  have hs : FP.Structural.evalOp M p (.intShiftRight .nil offsetOp (.constNat 32)) s =
      some (fun _ => offset >>> 32) := by
    rw [FP.Structural.eval_intShiftRight, ho]
    simp only [evalOp_unfold, bind, Option.bind, pure]
  rw [eval_low_mask M p _ _ s hs, Int.shiftRight_eq_div_pow]
  rfl

#axiomsClean eval_low_mask
#axiomsClean eval_high_mask

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedCounterOperands
