import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeValues

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
open VeriTile Triton
open Source
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure CarryBinding (A : Target.Ops α) (c : Config) (r : Fin 64)
    (a : Carries α c.D) (b : Target.State α c.D) : Prop where
  maximum : ∀ l, lower A c.fp8 c.D l (a.maximum r)=b.maximum l
  denominator : ∀ l, lower A c.fp8 c.D l (a.denominator r)=b.denominator l
  accumulator : ∀ d, lower A c.fp8 c.D (lane d.val) (a.accumulator (r,d,()))=b.accumulator d

theorem initial_binding (A : Target.Ops α) (c : Config) (r : Fin 64) :
    CarryBinding A c r (initial c.D) (NativeGraph.initial A c.D) := by
  constructor <;> intro i <;> simp [initial,NativeGraph.initial,lower,literal]

theorem denominator_mul (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (scale y : Term α) (n : Nat) (r : Fin 64) (l : Fin 4) :
    lower A c.fp8 c.D l (.mul ((accumulated c s scale n).denominator r) y)=
      A.mul (lower A c.fp8 c.D l ((accumulated c s scale n).denominator r))
        (lower A c.fp8 c.D l y) := by
  cases n <;> rfl

theorem accumulator_mul (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (scale y : Term α) (n : Nat) (r : Fin 64) (d : Fin c.D) (l : Fin 4) :
    lower A c.fp8 c.D l (.mul ((accumulated c s scale n).accumulator (r,d,())) y)=
      A.mul (lower A c.fp8 c.D l ((accumulated c s scale n).accumulator (r,d,())))
        (lower A c.fp8 c.D l y) := by
  cases n <;> rfl

theorem mma64_congr (A : Target.Ops α) (fp8 : Bool) (x y u v : Nat→α)
    (a b : α) (n : Nat) (hn : n≤4) (hs : a=b)
    (hx : ∀ j, j<64 → x j=u j) (hy : ∀ j, j<64 → y j=v j) :
    mmaFold A.mma16 (mmaCoordinate fp8 64) x y a n=
      mmaFold A.mma16 (mmaCoordinate fp8 64) u v b n := by
  induction n with
  | zero => exact hs
  | succ n ih =>
    rw [mmaFold,mmaFold,ih (by omega)]
    congr 1
    · funext i
      exact hx _ (coordinate64 fp8 ⟨n,by omega⟩ i)
    · funext i
      exact hy _ (coordinate64 fp8 ⟨n,by omega⟩ i)

theorem step_binding (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (n : Nat) (b : Target.State α c.D)
    (hb : CarryBinding A c r (accumulated c s (argument C arg) n) b) :
    CarryBinding A c r (advance c s (argument C arg) n (accumulated c s (argument C arg) n))
      (Target.step A C input arg c.fp8 c.causal c.N c.SP (rows s r) n b) := by
  constructor
  · intro l
    simp only [advance,Target.step,lower,Fin.isLt,↓reduceDIte,
      scores_binding A C arg c s r input h hr,hb.maximum]
  · intro l
    simp only [advance,Target.step,lower,Fin.isLt,↓reduceDIte,
      scores_binding A C arg c s r input h hr,hb.maximum,hb.denominator,NativeGraph.exp]
  · intro d
    simp only [advance,Target.step,lower,Fin.isLt,↓reduceDIte,
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

#axiomsClean initial_binding
#axiomsClean denominator_mul
#axiomsClean accumulator_mul
#axiomsClean step_binding
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
