import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16SourceSafetyLoads

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem epilogue_prefix_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (backend A P) bounds ((epilogue p).take 2) s := by
  apply pure_safe
  intro stmt member current
  simp only [epilogue,source,typedSurface,ComputeKernel.surfaceBody,
    List.take,List.drop,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals simp only [safeOp_unfold,true_and,and_true]

theorem epilogue_prefix_run (A : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α)
    (h : Invariant P p initial p.blocks (rounds P p initial p.blocks) current) :
    run (backend A P) ((epilogue p).take 2) current=some (epilogueStart P p initial current) := by
  simp [epilogue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,numeric,backend,h.start,h.ooffset,h.numerator,h.denominator,
    epilogueStart,normalized,outputPointer,TileShape.indexToList,Region.cast,ofFloat,toFloat,
    ComputeDType.eraseDType,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  rfl

theorem epilogue_store_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : ∀ i : QLane p,row p initial i.1<p.length → outputOffset p initial i<bounds p.output) :
    traceSafeList (backend A P) bounds ((epilogue p).drop 2) (epilogueStart P p initial current) := by
  rw [show (epilogue p).drop 2=[((epilogue p)[2]'(by change 2<3;decide))] from rfl,singleton_safe]
  simp [epilogue,source,typedSurface,ComputeKernel.surfaceBody,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,epilogueStart,outputPointer,Region.cast,TileShape.indexToList]
  intro m d active
  simpa [outputOffset,row,BlockPtr.address_2d_offsets,Nat.add_assoc] using backed (m,d,PUnit.unit) active

theorem epilogue_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (h : Invariant P p initial p.blocks (rounds P p initial p.blocks) current)
    (backed : ∀ i : QLane p,row p initial i.1<p.length → outputOffset p initial i<bounds p.output) :
    traceSafeList (backend A P) bounds (epilogue p) current := by
  rw [show epilogue p=(epilogue p).take 2++(epilogue p).drop 2 from rfl,traceSafeList_append]
  refine ⟨epilogue_prefix_safe A P p bounds current,?_⟩
  intro next executed
  have same := (epilogue_prefix_run A P p initial current h).symm.trans executed
  cases Option.some.inj same
  exact epilogue_store_safe A P p bounds initial current backed

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
