import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostStorage

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.HostIdentity
open VeriTile.Triton Storage Host
noncomputable section

/-- Nominal storage freshness includes empty tensors whose data pointers may
both be zero. Reshaping a result preserves the allocated storage identity. -/
theorem forward_fresh {α : Type} (r : Request α) (h : Call) (live : Caller)
    (p : Prepared h .forward live) (actual : CellHeap α) :
    ¬live.identities (outputTensor r h p actual).storageId :=
  p.freshIdentity "y" (by simp [buffers]) (by simp [newRole,outputs])

theorem gradients_fresh {α : Type} (ctx : Context α) (live : Caller)
    (p : BackwardPrepared ctx live) (actual : CellHeap α) :
    ¬live.identities (gradientTensors ctx p actual).1.storageId ∧
    ¬live.identities (gradientTensors ctx p actual).2.storageId ∧
    (gradientTensors ctx p actual).1.storageId≠(gradientTensors ctx p actual).2.storageId := by
  refine ⟨p.freshIdentity "dx" (by simp [buffers]) (by simp [newRole,outputs]),p.freshIdentity "dw" (by simp [buffers]) (by simp [newRole,outputs]),?_⟩
  exact p.storage.identities "dx" (by simp [buffers]) "dw" (by simp [buffers]) (by decide)

theorem saved_identity {α : Type} (ctx : Context α) (live : Caller) (p : BackwardPrepared ctx live)
    (r : RegionName) (saved : savedRole r) :
    (p.storage.allocation r (saved_backward r saved)).storageId=
      (ctx.storage.allocation r (saved_forward r saved)).storageId :=
  congrArg Native.Allocation.storageId (p.reuse r saved)

#axiomsClean forward_fresh
#axiomsClean gradients_fresh
#axiomsClean saved_identity
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.HostIdentity
