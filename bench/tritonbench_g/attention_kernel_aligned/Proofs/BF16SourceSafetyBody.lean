import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16SourceSafetyLoads

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem score_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (c : Nat) (st : Accumulator α p)
    (positive : 0<p.blockN) (h : Invariant P p initial c st current)
    (key : ∀ i : KLane p,c*p.blockN+i.2.1.val<p.length → kOffset p initial c i<bounds p.k)
    (value : ∀ i : VLane p,c*p.blockN+i.1.val<p.length → vOffset p initial c i<bounds p.v)
    (bias : ∀ i : Fin p.blockM,row p initial i<p.length → b0Offset p initial c i<bounds p.bias) :
    traceSafeList (backend A P) bounds (scorePart p)
      (current.setReg "start_n" .nat [] (fun _ => c*p.blockN)) := by
  have nonzero : p.blockN≠0 := by omega
  simp [scorePart,body,source,typedSurface,ComputeKernel.surfaceBody,traceSafeList_unfold,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,backend,
    h.memory,h.start,h.qoffset,h.ooffset,h.boffset,h.brows,h.bias,h.query,h.keyPtr,h.valuePtr,
    kPointer,vPointer,integralMod,natLt,Fin.isLt,boundaryZero,
    TileShape.indexToList,Region.cast,nonzero,ofFloat,toFloat,ComputeDType.eraseDType,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  exact ⟨by simpa [kOffset,BlockPtr.address_2d_offsets,Nat.add_assoc] using key,
    by simpa [vOffset,BlockPtr.address_2d_offsets,Nat.add_assoc] using value,
    by simpa [b0Offset,row,integralFloorDiv,nonzero,Nat.add_assoc] using bias⟩

theorem tail_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (backend A P) bounds (tailPart p) s := by
  apply pure_safe
  intro stmt member current
  simp only [tailPart,body,source,typedSurface,ComputeKernel.surfaceBody,
    List.drop,List.head?_cons,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
    simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem body_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (c : Nat) (st : Accumulator α p)
    (positive : 0<p.blockN) (h : Invariant P p initial c st current)
    (key : ∀ i : KLane p,c*p.blockN+i.2.1.val<p.length → kOffset p initial c i<bounds p.k)
    (value : ∀ i : VLane p,c*p.blockN+i.1.val<p.length → vOffset p initial c i<bounds p.v)
    (bias : ∀ i : Fin p.blockM,row p initial i<p.length → b0Offset p initial c i<bounds p.bias) :
    traceSafeList (backend A P) bounds (body p)
      (current.setReg "start_n" .nat [] (fun _ => c*p.blockN)) := by
  rw [show body p=scorePart p++tailPart p from rfl,traceSafeList_append]
  exact ⟨score_safe A P p bounds initial current c st positive h key value bias,
    fun next _ => tail_safe A P p bounds next⟩

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
