import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16SourceSafetyLoads

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem middle_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (backend A P) bounds (biasPrefix p) s := by
  apply pure_safe
  intro stmt member current
  simp only [biasPrefix,prologue,source,typedSurface,ComputeKernel.surfaceBody,
    List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl <;>
    simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem middle_run (A : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    run (backend A P) (biasPrefix p)
      ((prefixResult P p s).setReg "q" .bf16 [p.blockM,p.head] (qValue P p s))=
    some (middleResult P p s ((prefixResult P p s).setReg "q" .bf16 [p.blockM,p.head] (qValue P p s))) := by
  simp [biasPrefix,prologue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,backend,prefixResult,middleResult,qScaled,scaleValue,biasBase,
    integralMod,ofFloat,toFloat,ComputeDType.eraseDType,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  rfl

theorem bias_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (backed : ∀ i : ScoreLane p,row p s i.1<p.length → b1Offset p s i<bounds p.bias) :
    traceSafeList (backend A P) bounds (biasLoad p)
      (middleResult P p s ((prefixResult P p s).setReg "q" .bf16 [p.blockM,p.head] (qValue P p s))) := by
  rw [show biasLoad p=[((prologue p)[22]'(by change 22<23;decide))] from rfl,singleton_safe]
  simpa [prologue,source,typedSurface,ComputeKernel.surfaceBody,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,numeric,backend,prefixResult,middleResult,Region.cast,
    b1Offset,row,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,natLt,Nat.add_assoc]
    using backed

theorem prologue_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (query : ∀ i : QLane p,row p s i.1<p.length → qOffset p s i<bounds p.q)
    (bias : ∀ i : ScoreLane p,row p s i.1<p.length → b1Offset p s i<bounds p.bias) :
    traceSafeList (backend A P) bounds (prologue p) s := by
  rw [show prologue p=queryPrefix p++(queryLoad p++(biasPrefix p++biasLoad p)) from by
      rw [queryPrefix_source];rfl,
    traceSafeList_append]
  refine ⟨prefix_safe A P p bounds s,?_⟩
  intro first executed
  have first_eq := (prefix_run A P p s).symm.trans executed
  cases Option.some.inj first_eq
  rw [traceSafeList_append]
  refine ⟨query_safe A P p bounds s query,?_⟩
  intro second executed
  have second_eq := (query_run A P p s).symm.trans executed
  cases Option.some.inj second_eq
  rw [traceSafeList_append]
  refine ⟨middle_safe A P p bounds _,?_⟩
  intro third executed
  have third_eq := (middle_run A P p s).symm.trans executed
  cases Option.some.inj third_eq
  exact bias_safe A P p bounds s bias

#axiomsClean prologue_safe
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
