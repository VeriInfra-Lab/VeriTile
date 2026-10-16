import bench.tritonbench_g.triton_attention.Proofs.FPSoftware
import bench.tritonbench_g.triton_attention.Proofs.FPForwardReadback
import bench.tritonbench_g.triton_attention.Proofs.FPBackwardFrame
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
open AttentionKernelAligned.FPValue (ascendingPlans)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
local instance : DecidableEq (Cell Word) := by
  intro a b
  cases a with
  | mk da va =>
    cases b with
    | mk db vb =>
      by_cases h : da=db
      · subst db
        letI : DecidableEq (Value Word da) := by cases da <;> infer_instance
        exact decidable_of_iff (va=vb) (by simp)
      · exact isFalse (by intro he; cases he; exact h rfl)
/-- Explicit sequential software dot used only for the nonempty witnesses. It
is not identified with any native tensor-core or fused-accumulator schedule. -/
def witnessDot {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word) :
    TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc j => FP.add .ieeeValue acc
      (FP.mul .ieeeValue (a (i.1,j,PUnit.unit)) (b (j,i.2.1,PUnit.unit)))) (FP.Value.zero .fp32)
  | _::tail => fun i => witnessDot (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length
/-- Only exp(±0)=1 and exp(-Inf)=+0 occur below; the remaining inputs have an
arbitrary NaN extension. This does not provide a general exponential model. -/
def witnessExp (x : Word) : Word :=
  if x.isZero then ⟨0x3f800000⟩ else if x.bits=0xff800000 then FP.Value.zero .fp32 else FP.Value.qNaN .fp32
def witnessConvert (src dst : FloatDType) (x : Word) : Word :=
  match formatOf src,formatOf dst with
  | some a,some b => castWord a b x
  | _,_ => FP.Value.qNaN .fp32
def witnessExternal : External :=
  { hostLiteral := fun _ _ _ => FP.Value.qNaN .fp32
    exp := fun _ => witnessExp
    dot := fun _ left right out _ {batch} {m} {k} {n} a b => fun i => ofFloat out
      (witnessConvert .fp32 out (witnessDot (batch:=batch) (m:=m) (k:=k) (n:=n)
        (fun j => witnessConvert left .fp32 (toFloat left (a j)))
        (fun j => witnessConvert right .fp32 (toFloat right (b j))) i))
    otherCast := fun _ _ _ _ => FP.Value.qNaN .fp32 }
def witnessPre : PreParameters :=
  { dtype:=.fp16, output:="Out", gradient:="DO", denominator:="L",newGradient:="NewDO",delta:="Delta",rows:=1,head:=2 }
def witnessPreState : State Word :=
  { mem := fun r o => if r="Out" then .mk .fp16 ⟨0x3c00⟩ else
      if r="DO" then .mk .fp16 (if o=0 then ⟨0x4400⟩ else ⟨0x4600⟩) else
      if r="L" then .mk .fp32 ⟨0x40000000⟩ else .mk .nat 73
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d }
def witnessPreLane0 : PreLane witnessPre := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)
def witnessPreLane1 : PreLane witnessPre := (⟨0,by decide⟩,⟨1,by decide⟩,PUnit.unit)
def witnessPreRow : PreRow witnessPre := (⟨0,by decide⟩,PUnit.unit)
theorem preprocess_witness_values :
    preGradientCell (software witnessExternal ascendingPlans) witnessPre witnessPreState witnessPreLane0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
    preGradientCell (software witnessExternal ascendingPlans) witnessPre witnessPreState witnessPreLane1=.mk .fp16 (⟨0x4200⟩ : Word) ∧
    preDeltaCell (software witnessExternal ascendingPlans) witnessPre witnessPreState witnessPreRow=.mk .fp32 (⟨0x40a00000⟩ : Word) := by
  refine ⟨?_,?_,?_⟩
  all_goals apply congrArg (Cell.mk _)
  all_goals change (_ : Word) = (_ : Word)
  all_goals delta preGradientCell preDeltaCell preDelta preProducts preQuotient preInput preDenominator
  all_goals norm_num [software, convert, witnessPre, formatOf]
  all_goals decide +kernel

def witnessForward : ForwardParameters :=
  { dtype:=.fp16,q:="Q",k:="K",v:="V",denominator:="L",maximum:="M",output:="Out",scale:=1,
    headStride:=2,context:=2,rows:=2,blockM:=1,blockN:=1,head:=1 }
def witnessForwardState : State Word :=
  { mem := fun r o => if r="Q" ∨ r="K" then .mk .fp16 ⟨0⟩ else
      if r="V" then .mk .fp16 (if o=0 then ⟨0x3c00⟩ else ⟨0x4000⟩) else .mk .nat 73
    regs := fun _ _ _ => none
    pids := fun i => if i=0 then 1 else 0
    numPids := fun _ => 2
    undef := fun d _ _ => defaultValue d }
def witnessForwardLane : FQuery witnessForward := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)
def witnessForwardRow : FRow witnessForward := (⟨0,by decide⟩,PUnit.unit)
theorem forward_witness_values :
    forwardOutput (software witnessExternal ascendingPlans) witnessForward witnessForwardState witnessForwardLane=.mk .fp16 (⟨0x3e00⟩ : Word) ∧
    forwardLCell (software witnessExternal ascendingPlans) witnessForward witnessForwardState witnessForwardRow=.mk .fp32 (⟨0x40000000⟩ : Word) ∧
    forwardMCell (software witnessExternal ascendingPlans) witnessForward witnessForwardState witnessForwardRow=.mk .fp32 (⟨0⟩ : Word) := by
  refine ⟨?_,?_,?_⟩
  all_goals apply congrArg (Cell.mk _)
  all_goals change (_ : Word) = (_ : Word)
  all_goals norm_num [forwardBlocks,forwardStop,witnessForwardState,forwardOutput,forwardLCell,forwardMCell,forwardRounds,forwardAdvance,forwardValueDot,
    forwardHalfWeights,forwardNormalized,forwardRescaled,forwardReciprocal,forwardDenominator,forwardWeights,
    forwardOldDenominator,forwardMax,forwardScore,forwardInitial,forwardZero,software,witnessForward]
  all_goals try delta forwardValueDot forwardHalfWeights forwardNormalized forwardRescaled forwardReciprocal forwardDenominator forwardWeights forwardOldDenominator forwardMax forwardScore forwardZero
  all_goals norm_num [witnessForward]
  all_goals decide +kernel

def witnessBackward : BackwardParameters :=
  { dtype:=.fp16,q:="Q",k:="K",v:="V",gradient:="DO",dq:="DQ",dk:="DK",dv:="DV",maximum:="M",delta:="Delta",scale:=1,
    batchStride:=2,headStride:=2,heads:=1,rows:=2,blockM:=1,head:=1,blocks:=2 }
def witnessBackwardState : State Word :=
  { mem := fun r o => if r="Q" ∨ r="K" then .mk .fp16 ⟨0⟩ else
      if r="V" then .mk .fp16 (if o=0 then ⟨0x3c00⟩ else ⟨0x4000⟩) else
      if r="DO" then .mk .fp16 ⟨0x3c00⟩ else
      if r="M" ∨ r="Delta" ∨ r="DQ" then .mk .fp32 ⟨0⟩ else .mk .nat 73
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d }
theorem backward_witness_values :
    backwardFinalMemory (software witnessExternal ascendingPlans) witnessBackward witnessBackwardState "DQ" 0=.mk .fp32 (⟨0⟩ : Word) ∧
    backwardFinalMemory (software witnessExternal ascendingPlans) witnessBackward witnessBackwardState "DQ" 1=.mk .fp32 (⟨0⟩ : Word) ∧
    backwardFinalMemory (software witnessExternal ascendingPlans) witnessBackward witnessBackwardState "DK" 0=.mk .fp16 (⟨0⟩ : Word) ∧
    backwardFinalMemory (software witnessExternal ascendingPlans) witnessBackward witnessBackwardState "DV" 0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
    backwardFinalMemory (software witnessExternal ascendingPlans) witnessBackward witnessBackwardState "DV" 1=.mk .fp16 (⟨0x3c00⟩ : Word) := by
  norm_num [backwardFinalMemory,backwardRounds,backwardOuterAdvance,backwardInnerRounds,backwardInnerInitial,backwardInnerAdvance,backwardQueryCell,
    backwardQueryValue,backwardQueryDot,backwardKeyDot,backwardValueDot,backwardDS,backwardDP,backwardDPInitial,
    backwardDPDot,backwardWeights,backwardMaskedQK,software,witnessBackward]
  all_goals try delta backwardQueryValue backwardQueryDot backwardKeyDot backwardValueDot backwardDS backwardDP backwardDPInitial backwardDPDot backwardWeights backwardMaskedQK
  all_goals norm_num [witnessBackward]
  all_goals decide +kernel

/-- Nonempty two-column preprocess with actual half inputs/output and fp32
row metadata. This is a supporting instance of the arbitrary-dimension theorem. -/
theorem preprocess_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (preSource witnessPre) witnessPreState=some final ∧
      final.mem "NewDO" 0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
      final.mem "NewDO" 1=.mk .fp16 (⟨0x4200⟩ : Word) ∧
      final.mem "Delta" 0=.mk .fp32 (⟨0x40a00000⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := by
  have hi : Function.Injective (preOffset witnessPre witnessPreState) := by
    intro a b he
    have ha : a.1.val < 1 := a.1.isLt
    have hb : b.1.val < 1 := b.1.isLt
    apply Prod.ext
    · apply Fin.ext; omega
    · apply Prod.ext
      · apply Fin.ext
        simpa [preOffset,witnessPre,witnessPreState] using he
      · exact Subsingleton.elim _ _
  obtain ⟨final,he,_,hg,hd,hframe⟩ := preprocessCorrectness rest (software witnessExternal ascendingPlans)
    witnessPre witnessPreState (by decide) hi
  refine ⟨final,he,?_,?_,?_,?_⟩
  · have h := hg witnessPreLane0
    rw [preprocess_witness_values.1] at h
    simpa [preOffset,witnessPre,witnessPreState,witnessPreLane0] using h
  · have h := hg witnessPreLane1
    rw [preprocess_witness_values.2.1] at h
    simpa [preOffset,witnessPre,witnessPreState,witnessPreLane1] using h
  · have h := hd witnessPreRow
    rw [preprocess_witness_values.2.2] at h
    simpa [preRow,witnessPre,witnessPreState,witnessPreRow] using h
  · simpa [witnessPreState] using hframe "other" 7 (Or.inl (by decide)) (Or.inl (by decide))

/-- Two KV blocks, not a single-iteration specialization of the main theorem. -/
theorem forward_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (forwardSource witnessForward) witnessForwardState=some final ∧
      final.mem "Out" 1=.mk .fp16 (⟨0x3e00⟩ : Word) ∧
      final.mem "L" 1=.mk .fp32 (⟨0x40000000⟩ : Word) ∧
      final.mem "M" 1=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := by
  obtain ⟨final,he,_,ho,hl,hm,hframe⟩ := forwardCorrectness rest (software witnessExternal ascendingPlans)
    witnessForward witnessForwardState (by decide) (by decide) (by decide) (by decide) (by decide)
    (by intro a b _; exact @Subsingleton.elim (Fin 1 × Fin 1 × PUnit) inferInstance a b)
  refine ⟨final,he,?_,?_,?_,?_⟩
  · have h := ho witnessForwardLane (by decide)
    rw [forward_witness_values.1] at h
    simpa [forwardOffset,forwardHeadRow,forwardQueryRow,witnessForward,witnessForwardState,witnessForwardLane] using h
  · have h := hl witnessForwardRow
    rw [forward_witness_values.2.1] at h
    simpa [forwardRowOffset,forwardQueryRow,witnessForward,witnessForwardState,witnessForwardRow] using h
  · have h := hm witnessForwardRow
    rw [forward_witness_values.2.2] at h
    simpa [forwardRowOffset,forwardQueryRow,witnessForward,witnessForwardState,witnessForwardRow] using h
  · simpa [witnessForwardState] using hframe "other" 7 (Or.inl (by decide)) (Or.inl (by decide)) (Or.inl (by decide))

/-- Two outer key blocks and three inner query iterations. Every stored result
is derived from initial memory, including cumulative DQ load/add/store. -/
theorem backward_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (backwardSource witnessBackward) witnessBackwardState=some final ∧
      final.mem "DQ" 0=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "DQ" 1=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "DK" 0=.mk .fp16 (⟨0⟩ : Word) ∧
      final.mem "DV" 0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
      final.mem "DV" 1=.mk .fp16 (⟨0x3c00⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := by
  obtain ⟨final,he,hm,_,hframe⟩ := backwardCorrectness rest (software witnessExternal ascendingPlans)
    witnessBackward witnessBackwardState (by decide)
  refine ⟨final,he,?_,?_,?_,?_,?_,?_⟩
  · rw [hm]; exact backward_witness_values.1
  · rw [hm]; exact backward_witness_values.2.1
  · rw [hm]; exact backward_witness_values.2.2.1
  · rw [hm]; exact backward_witness_values.2.2.2.1
  · rw [hm]; exact backward_witness_values.2.2.2.2
  · simpa [witnessBackwardState] using hframe "other" 7 (by simp [backwardWritten,witnessBackward])

#axiomsClean preprocess_bit_success
#axiomsClean forward_bit_success
#axiomsClean backward_bit_success

#axiomsClean preprocess_witness_values
#axiomsClean forward_witness_values
#axiomsClean backward_witness_values
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
