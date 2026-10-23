import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryExecution
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeLoop

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryNative
open VeriTile Triton Source Native
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem denominator_mul (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (scale y : Term α) (n : Nat) (r : Fin 64) (l : Fin 4) :
    lower A c.fp8 c.D l (.mul ((CarrySource.accumulated c s scale n).denominator r) y)=
      A.mul (lower A c.fp8 c.D l ((CarrySource.accumulated c s scale n).denominator r))
        (lower A c.fp8 c.D l y) := by
  cases n <;> rfl

theorem accumulator_mul (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (scale y : Term α) (n : Nat) (r : Fin 64) (d : Fin c.D) (l : Fin 4) :
    lower A c.fp8 c.D l (.mul ((CarrySource.accumulated c s scale n).accumulator (r,d,())) y)=
      A.mul (lower A c.fp8 c.D l ((CarrySource.accumulated c s scale n).accumulator (r,d,())))
        (lower A c.fp8 c.D l y) := by
  cases n <;> rfl

theorem step_binding (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (n : Nat) (b : Target.State α c.D)
    (hb : CarryBinding A c r (CarrySource.accumulated c s (argument C arg) n) b) :
    CarryBinding A c r (CarrySource.advance c s (argument C arg) n (CarrySource.accumulated c s (argument C arg) n))
      (Target.step A C input arg c.fp8 c.causal c.N c.SP (rows s r) n b) := by
  constructor
  · intro l
    simp only [CarrySource.advance,Target.step,lower,Fin.isLt,↓reduceDIte,
      scores_binding A C arg c s r input h hr,hb.maximum]
  · intro l
    simp only [CarrySource.advance,Target.step,lower,Fin.isLt,↓reduceDIte,
      scores_binding A C arg c s r input h hr,hb.maximum,hb.denominator,NativeGraph.exp]
  · intro d
    simp only [CarrySource.advance,Target.step,lower,Fin.isLt,↓reduceDIte,
      scores_binding A C arg c s r input h hr,hb.maximum,hb.denominator,
      denominator_mul,accumulator_mul,hb.accumulator,NativeGraph.exp,literal,
      show (1:ℝ)≠0 by norm_num,↓reduceIte]
    apply mma64_congr A c.fp8 _ _ _ _ _ _ 4 (by omega)
    · rw [accumulator_mul A c s (argument C arg)]
      simp only [denominator_mul A c s (argument C arg),lower,Fin.isLt,↓reduceDIte,
        scores_binding A C arg c s r input h hr,hb.maximum,hb.denominator,hb.accumulator,
        NativeGraph.exp,literal,show (1:ℝ)≠0 by norm_num,↓reduceIte]
    · intro j hj
      simp only [hj,↓reduceDIte,column,Nat.mod_eq_of_lt hj]
    · intro j hj
      simp only [hj,↓reduceDIte,column,Nat.mod_eq_of_lt hj,
        value_binding A c s r input h]

#axiomsClean denominator_mul
#axiomsClean accumulator_mul
#axiomsClean step_binding
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarryNative
