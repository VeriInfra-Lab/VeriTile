import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16SourceSafetySteps

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem singleton_safe (A : Algebra α) (bounds : RegionName → Nat)
    (stmt : ComputeStmt) (s : State α) :
    traceSafeList A bounds [stmt] s ↔ stepSafe A bounds stmt s := by
  rw [traceSafeList_cons]
  simp

theorem query_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (backed : ∀ i : QLane p,row p s i.1<p.length → qOffset p s i<bounds p.q) :
    traceSafeList (backend A P) bounds (queryLoad p) (prefixResult P p s) := by
  rw [show queryLoad p=[((prologue p)[15]'(by change 15<23;decide))] from rfl,singleton_safe]
  simp [prologue,source,typedSurface,ComputeKernel.surfaceBody,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,prefixResult,Region.cast,qPointer,TileShape.indexToList]
  intro m d active
  simpa [qOffset,qPointer,row,BlockPtr.address_2d_offsets,Nat.add_assoc] using backed (m,d,PUnit.unit) active

theorem query_run (A : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    run (backend A P) (queryLoad p) (prefixResult P p s)=
      some ((prefixResult P p s).setReg "q" .bf16 [p.blockM,p.head] (qValue P p s)) := by
  simp [queryLoad,prologue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,backend,prefixResult,qPointer,TileShape.indexToList,
    boundaryZero,ofFloat,toFloat,ComputeDType.eraseDType,Region.cast,qValue,qOffset,row,
    BlockPtr.address_2d_offsets,Nat.add_assoc]
  delta qValue qOffset row
  simp only [Nat.add_assoc]

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
