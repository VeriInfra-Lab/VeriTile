import bench.tritonbench_g.attention_kernel_aligned.FPCorrect
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- One explicit unfused software dot instance for execution witnesses. It
widens half inputs, multiplies/adds in fp32 left-to-right, then narrows only
the half-output dot. This is not a universal tensor-core schedule claim. -/
def witnessDot {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word) :
    TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc j => FP.add .ieeeValue acc
      (FP.mul .ieeeValue (widenHalf (a (i.1,j,PUnit.unit)))
        (widenHalf (b (j,i.2.1,PUnit.unit))))) (FP.Value.zero .fp32)
  | _::tail => fun i => witnessDot (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

/-- Only exp2(±0)=1 and exp2(-Inf)=+0 are reached by this concrete witness.
Other inputs deliberately retain an arbitrary total NaN extension; no global
exp2 implementation or correctness assertion is attached to it. -/
def witnessExp2 (x : Word) : Word :=
  if x.isZero then ⟨0x3f800000⟩ else
  if x.bits=0xff800000 then FP.Value.zero .fp32 else FP.Value.qNaN .fp32

def witnessExternal : External :=
  { hostLiteral := fun r => if r=1 then ⟨0x3f800000⟩ else FP.Value.qNaN .fp32
    exp2 := witnessExp2
    dot := fun _ left right out _ {batch} {m} {k} {n} a b =>
      fun i => ofFloat out (if out=.fp16 then narrowHalf
        (witnessDot (batch:=batch) (m:=m) (k:=k) (n:=n)
          (fun j => toFloat left (a j)) (fun j => toFloat right (b j)) i)
        else witnessDot (batch:=batch) (m:=m) (k:=k) (n:=n)
          (fun j => toFloat left (a j)) (fun j => toFloat right (b j)) i) }

def witnessParameters : Parameters :=
  { q:="Q", k:="K", v:="V", bias:="B", output:="Q", scale:=1,
    headStride:=2, biasHeadStride:=4, blockM:=1, blockN:=1, head:=1,
    biasLast:=1, biasRowStride:=2, blocks:=2 }
def witnessState : State Word :=
  { mem := fun r o => if r="Q" then .mk .fp16 ⟨0x3c00⟩ else
      if r="K" then .mk .fp16 ⟨0⟩ else
      if r="V" then .mk .fp16 (if o=0 then ⟨0x3c00⟩ else ⟨0x4000⟩) else
      if r="B" then .mk .fp16 ⟨0⟩ else .mk .nat 73
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d }
def witnessLane : QLane witnessParameters := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

theorem witness_values :
    qScaled (software witnessExternal ascendingPlans) witnessParameters witnessState witnessLane = ⟨0x3dc5⟩ ∧
    outputValue (software witnessExternal ascendingPlans) witnessParameters witnessState witnessLane = ⟨0x3e00⟩ := by
  norm_num [qScaled,scaleValue,outputValue,rounds,advance,weightedDot,weights,alphaValue,
    newMax,score32,score,qkDot,halfZero,initialAccumulator,software,witnessParameters,witnessExternal]
  all_goals delta weightedDot weights alphaValue newMax score32 score qkDot qScaled scaleValue
  all_goals norm_num [witnessParameters,halfZero]
  all_goals decide +kernel

/-- Actual two-block execution with output aliasing the initial query region.
The independent result averages the two value blocks. Frame covers every
unwritten query cell and another region. This supports, rather than limits,
the parameterized contracts. -/
theorem alias_two_block_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (source witnessParameters) witnessState=some final ∧
      final.mem "Q" 0=.mk .fp16 (⟨0x3e00⟩ : Word) ∧
      final.mem "Q" 1=.mk .fp16 (⟨0x3c00⟩ : Word) ∧
      final.mem "V" 1=.mk .fp16 (⟨0x4000⟩ : Word) ∧
      final.mem "other" 0=.mk .nat 73 := by
  obtain ⟨final,he,_,ho,hframe⟩ := FPCorrect.attention_aligned_full_scalar_bit_correctness rest
    witnessExternal ascendingPlans witnessParameters witnessState (by decide) (by decide) (by decide) (by decide)
  refine ⟨final,he,?_,?_,?_,?_⟩
  · have h := ho witnessLane
    rw [witness_values.2] at h
    simpa only [witnessParameters,qOffset,headBase,row,witnessState,witnessLane,
      Nat.zero_mul,Nat.zero_add,Nat.add_zero,Nat.mul_one] using h
  · have h := hframe "Q" 1 (Or.inr (by intro i; have := i.1.isLt; have := i.2.1.isLt; simp [qOffset,row,headBase,witnessParameters,witnessState]))
    simpa [witnessState] using h
  · have h := hframe "V" 1 (Or.inl (by decide))
    simpa [witnessState] using h
  · have h := hframe "other" 0 (Or.inl (by decide))
    simpa [witnessState] using h

#axiomsClean witness_values
#axiomsClean alias_two_block_execution
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
