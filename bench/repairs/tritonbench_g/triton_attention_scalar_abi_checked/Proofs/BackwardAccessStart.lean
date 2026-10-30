import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.AccessGeometry
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardOuterStart

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

theorem backward_prologue_safe (M : Algebra α) (p : BackwardParameters)
    (s : State α) (bounds : RegionName→Nat) :
    Safety.traceSafeList M bounds (backwardPrologue p) s := by
  apply AccessSafety.list_safe_of_steps
  intro statement member state
  change statement∈[_,_,_,_,_,_,_,_,_,_,_,_,_] at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem backward_start_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block : Nat) (mem : Memory α)
    (inv : BackwardEntryInvariant p initial block mem current) (bounds : RegionName→Nat)
    (keys : ∀ i : BLane p,backwardOffset p initial block i<bounds p.k)
    (values : ∀ i : BLane p,backwardOffset p initial block i<bounds p.v) :
    Safety.traceSafeList (backend rest P) bounds ((backwardOuterBody p).take 12)
      (current.setReg "start_n" .nat [] (fun _=>block)) := by
  simp [backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    inv.hz,inv.dqBase,inv.keyPtr,inv.valuePtr,backwardPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  exact ⟨fun i d _=>keys (i,d,PUnit.unit),fun i d _=>values (i,d,PUnit.unit)⟩

#axiomsClean backward_start_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
