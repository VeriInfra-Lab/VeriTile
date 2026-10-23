import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Domain
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Copy

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000
def helperCode (reductionAxis : Int) (temporary final : QuantBuffers) (v : QuantInvocation G K) : Code I F H Unit := do
  let _ ← quantGridCode temporary v.plan v.groups
  if reductionAxis=0 then copyCode temporary.dataAddress final.dataAddress (quantCells G K) else pure ()
def helperRun (a : Ops I F H) (reductionAxis : Int) (temporary final : QuantBuffers)
    (v : QuantInvocation G K) (m : Memory F H) :=
  if reductionAxis=0 then axis0Helper a temporary final v m else quantRun a temporary v m
theorem eval_helperCode (a : Ops I F H) (reductionAxis : Int) (temporary final : QuantBuffers)
    (v : QuantInvocation G K) (m : Memory F H) :
    (helperCode reductionAxis temporary final v).eval a m=((),helperRun a reductionAxis temporary final v m) := by
  unfold helperCode helperRun
  simp only [bind,eval_bind,Prod.snd]
  by_cases h : reductionAxis=0
  · simp only [if_pos h,axis0Helper,copyRun,quantRun]
  · simp only [if_neg h,pure,eval_pure,quantRun]

structure HelperAdmission (rows cols G K : Nat) (reductionAxis : Int) (temporary final : QuantBuffers) : Prop where
  axes : reductionAxis=0 ∨ reductionAxis=1 ∨ reductionAxis= -1 ∨ reductionAxis= -2
  reductionNonempty : 0<K
  contiguousInput : (reductionAxis=0 ∨ reductionAxis= -2) → G=cols ∧ K=rows ∧ temporary.inputGroupStride=1 ∧ temporary.inputElementStride=cols
  rowInput : (reductionAxis=1 ∨ reductionAxis= -1) → G=rows ∧ K=cols ∧ temporary.inputGroupStride=cols ∧ temporary.inputElementStride=1
  temporaryLayout : temporary.transposeOutput=decide (reductionAxis=0 ∨ reductionAxis= -2)
  finalLayout : reductionAxis=0 → final.transposeOutput=Bool.false
  scales : temporary.scalePrecision=.single
  initial : temporary.initialMaximumZero=Bool.false
  separate : temporary.Separate
  copySeparate : reductionAxis=0 → temporary.output≠final.output ∧ temporary.scales≠final.output
  noCopyIdentity : reductionAxis≠0 → final=temporary
structure BoundHelper (I F H : Type) (rows cols G K : Nat) (reductionAxis : Int) (temporary final : QuantBuffers) where
  invocation : QuantInvocation G K
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  admitted : HelperAdmission rows cols G K reductionAxis temporary final
  lowered : Code I F H Unit
  sourceLowering : lowered=helperCode reductionAxis temporary final invocation
def compiledHelper (abi : LocalEventABI I F H σ a)
    (call : BoundHelper I F H rows cols G K reductionAxis temporary final) (state : σ) : σ :=
  (call.lowered.native abi state).2
theorem compiledHelper_refines (abi : LocalEventABI I F H σ a)
    (call : BoundHelper I F H rows cols G K reductionAxis temporary final) (state : σ) :
    abi.observe (compiledHelper abi call state)=helperRun a reductionAxis temporary final call.invocation (abi.observe state) := by
  unfold compiledHelper
  rw [call.sourceLowering]
  have h := congrArg Prod.snd ((helperCode reductionAxis temporary final call.invocation).native_refines abi state)
  simpa only [Prod.snd,eval_helperCode] using h
theorem helper_value (a : Ops I F H) (reductionAxis : Int) (temporary final : QuantBuffers)
    (admit : HelperAdmission rows cols G K reductionAxis temporary final) (v : QuantInvocation G K)
    (m : Memory F H) (x : QuantCell G K) :
    helperRun a reductionAxis temporary final v m (final.dataAddress x)=.inl (quantValue a temporary v.plan m x) := by
  unfold helperRun
  split_ifs with h
  · exact axis0Helper_at a temporary final admit.separate (admit.copySeparate h).1 v m x
  · rw [admit.noCopyIdentity h]
    exact quantRun_at a temporary admit.separate v m (.inl x)
theorem helper_scale (a : Ops I F H) (reductionAxis : Int) (temporary final : QuantBuffers)
    (admit : HelperAdmission rows cols G K reductionAxis temporary final) (v : QuantInvocation G K)
    (m : Memory F H) (g : Fin G) :
    helperRun a reductionAxis temporary final v m (temporary.scaleAddress g)=
      .inr (.inl (quantScale a temporary v.plan m g)) := by
  have scale := quantRun_at a temporary admit.separate v m (.inr g)
  simp only [QuantBuffers.writeAddress,quantExpected,admit.scales] at scale
  unfold helperRun
  split_ifs with h
  · unfold axis0Helper
    rw [copyRun_frame a temporary final (admit.copySeparate h).1]
    · exact scale
    · intro x eq
      exact (admit.copySeparate h).2 (congrArg Prod.fst eq)
  · exact scale
/-- Python tuple ABI: values, squeezed scale, and literal None. Metadata and
fresh allocations are ordinary Tensor contracts, outside scalar numeric leaves. -/
def helperResult (values scales : TensorHandle) (state : σ) : TensorHandle × TensorHandle × Option Unit × σ :=
  (values,scales,none,state)
theorem helper_returns (values scales : TensorHandle) (state : σ) :
    (helperResult values scales state).2.2.1=none := rfl
#axiomsClean compiledHelper_refines
#axiomsClean helper_value
#axiomsClean helper_scale
#axiomsClean helper_returns
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
