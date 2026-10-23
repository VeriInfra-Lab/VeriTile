import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeForwardStep
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

theorem forwardNormalized_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) (j : Fin 128) :
    lowerScalar one .forward ((modelScalar one .forward).cast (some .fp32) .fp16 .fp32
      (forwardHalfWeights (modelScalar one .forward) (forwardParameters c scale) s block st (row,j,())) (Reduction.owner j.val).val)=
      Target.normalized c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block a j.val := by
  change Target.half (.mul
    (lowerScalar one .forward (forwardWeights (modelScalar one .forward) (forwardParameters c scale) s block st (row,j,()) (Reduction.owner j.val).val))
    (.div (lowerScalar one .forward (literal 1))
      (lowerScalar one .forward (forwardDenominator (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) (Reduction.owner j.val).val))))=_
  rw [forwardWeights_native c scale one s x head block hq hkv st row col a ha,
    forwardDenominator_native c scale one s x head block hq hkv st row col a ha,replica_val]
  simp only [show literal 1=Stage.one from by simp [literal],lowerScalar]
  rfl

theorem forwardValueDot_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) (u : Nat) :
    forwardValueDot (modelScalar one .forward) (forwardParameters c scale) s block st (row,col,()) u=
      Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
        (Target.pad 128 (Target.normalized c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block a))
        (Target.pad 128 (fun j=>x.v head (block*128+j) col.val)) := by
  unfold forwardValueDot
  change Stage.dot (TritonAttention.Whole001.Native.width c.dim) 128
    (fun d=>if hd : d<128 then lowerScalar one .forward
      ((modelScalar one .forward).cast (some .fp32) .fp16 .fp32
        (forwardHalfWeights (modelScalar one .forward) (forwardParameters c scale) s block st (row,⟨d,hd⟩,()))
        (Reduction.owner d).val) else .word 0)
    (fun d=>if hd : d<128 then lowerScalar one .forward
      (forwardKV (forwardParameters c scale) s "V" block (⟨d,hd⟩,col,()) (Reduction.owner col.val).val) else .word 0)=_
  congr 1
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      exact forwardNormalized_native c scale one s x head block hq hkv st row col a ha ⟨j,hj⟩
    · simp only [Target.pad,dif_neg hj,if_neg hj]
  · funext j
    by_cases hj : j<128
    · simp only [Target.pad,dif_pos hj,if_pos hj]
      rw [hkv.2 (⟨j,hj⟩,col,())]
      rfl
    · simp only [Target.pad,dif_neg hj,if_neg hj]

theorem forwardAdvance_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) :
    AccAt c scale one (forwardAdvance (modelScalar one .forward) (forwardParameters c scale) s block st) row col
      (Target.advance c one scale x head (forwardQueryRow (forwardParameters c scale) s row) col.val block a) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro u
    exact forwardMax_native c scale one s x head block hq hkv st row col a ha u
  · intro u
    exact forwardDenominator_native c scale one s x head block hq hkv st row col a ha u
  · intro u
    change lowerScalar one .forward (.add
      (forwardRescaled (modelScalar one .forward) (forwardParameters c scale) s block st (row,col,()) u)
      (forwardValueDot (modelScalar one .forward) (forwardParameters c scale) s block st (row,col,()) u))=_
    rw [forwardValueDot_native c scale one s x head block hq hkv st row col a ha]
    change dot (TritonAttention.Whole001.Native.width c.dim) 128 _ _ Bool.true
      (.mul (lowerScalar one .forward (st.numerator (row,col,()) u))
        (.mul (.mul (lowerScalar one .forward (st.denominator (row,()) u))
          (Target.exp (lowerScalar one .forward (.sub (st.maximum (row,()) u)
            (forwardMax (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u)))))
          (.div (lowerScalar one .forward (literal 1))
            (lowerScalar one .forward (forwardDenominator (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u)))))=_
    rw [ha.2.2.1,ha.2.1,ha.2.2.2,ha.1,
      forwardMax_native c scale one s x head block hq hkv st row col a ha,
      forwardDenominator_native c scale one s x head block hq hkv st row col a ha]
    simp only [show literal 1=Stage.one from by simp [literal],lowerScalar]
    rfl
  · intro u b
    rfl

theorem forwardInitial_native (c : Shape) (scale : ℝ) (one : Bool) (row : Fin 128) (col : Fin c.dim) :
    AccAt c scale one (forwardInitial (modelScalar one .forward) (forwardParameters c scale)) row col Target.initial := by
  refine ⟨?_,?_,?_,?_⟩ <;> intros <;>
    simp [AccAt,forwardInitial,forwardZero,modelScalar,algebraScalar,algebra,literal,lowerScalar,Target.initial]

theorem forwardRounds_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head : Nat) (hq : ReadyQ c scale s x head)
    (row : Fin 128) (col : Fin c.dim) (n : Nat)
    (hkv : ∀ b,b<n → ReadyKV c scale s x head b) :
    AccAt c scale one (forwardRounds (modelScalar one .forward) (forwardParameters c scale) s n) row col
      (Target.rounds c one scale x head (forwardQueryRow (forwardParameters c scale) s row) col.val n) := by
  induction n with
  | zero=>exact forwardInitial_native c scale one row col
  | succ n ih=>
    exact forwardAdvance_native c scale one s x head n hq (hkv n (by omega))
      _ row col _ (ih (by intro b hb;exact hkv b (by omega)))

#axiomsClean forwardRounds_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
