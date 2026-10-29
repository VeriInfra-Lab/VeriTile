import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.NativeOperands
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Native
open VeriTile Triton FP.Structural Source Whole Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Source QK operands and the actual half bias accumulator feed the four
ordered m16n8k16 fragments. This is a staging identity, not a numerical law
equating a separately rounded dot/add with a seeded dot. -/
theorem qk_seed (p : Parameters) (s : State Expr) (c : Nat)
    (i : ScoreLane p) (u : Fin 4) :
    lower (rawScore (modelFor p) p s c i u)=
      dotHalf16 p.head i.1.val i.2.1.val
        (fun d=>lower (qScaled (modelFor p) p s (i.1,d,()) (physicalOwner d.val)))
        (fun d=>lower (kValue (modelFor p) p s c (d,i.2.1,()) (physicalOwner i.2.1.val)))
        (lower (.halfAdd (b0Value (modelFor p) p s c i.1 u)
          (b1Value (modelFor p) p s i u))) := by
  simp [rawScore,halfZero,qkDot,modelFor,model,binaryFor,algebra,dotValues,ofFloat,toFloat,literal,narrow,lower]

/-- Source probability/value operands and the separately rounded rescaled
accumulator feed the actual FP32 MMA seed. -/
theorem pv_seed (p : Parameters) (s : State Expr) (c : Nat)
    (st : Accumulator Expr p) (i : QLane p) (u : Fin 4) :
    lower ((advance (modelFor p) p s c st).numerator i u)=
      dotHalf32 p.blockN i.1.val i.2.1.val
        (fun j=>lower (narrow (weights (modelFor p) p s c st (i.1,j,()) (physicalOwner j.val))))
        (fun j=>lower (vValue (modelFor p) p s c (j,i.2.1,()) (physicalOwner i.2.1.val)))
        (.mul32 (lower (st.numerator i u))
          (lower (alphaValue (modelFor p) p s c st (i.1,()) u))) := by
  simp [advance,weightedDot,modelFor,model,binaryFor,algebra,dotValues,ofFloat,toFloat,lower]

/-- The online normalizer is one scalar FMA after the selected explicit
reduction; it is never replaced by a whole-reduction numerical premise. -/
theorem normalizer_fma (p : Parameters) (s : State Expr) (c : Nat)
    (st : Accumulator Expr p) (i : Row p) (u : Fin 4) :
    lower ((advance (modelFor p) p s c st).denominator i u)=
      .fma32 (lower (st.denominator i u))
        (lower (alphaValue (modelFor p) p s c st i u))
        (lower (((modelFor p).sum (some .fp32) ⟨1,by simp⟩ Bool.false
          (weights (modelFor p) p s c st) i) u)) := by
  simp [advance,modelFor,model,binaryFor,algebra,reductionFor,lower]

theorem bias_seed_bytes (a b : Word) :
    lower (.halfAdd (.ready (.bits a)) (.ready (.bits b)))=.add16 (.bits b) (.bits a) := rfl

#axiomsClean bias_seed_bytes
#axiomsClean qk_seed
#axiomsClean pv_seed
#axiomsClean normalizer_fma
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Native
