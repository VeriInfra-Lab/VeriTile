import bench.tritonbench_g.log_softmax.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LogSoftmax.KernelSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem active_bound (M N K BM C : Nat) (s : State A) (i : TileIndex [BM,C])
    (active : s.pids 0*BM+i.1.val<M ∧ i.2.1.val<N ∧ s.pids 1<K) :
    ((if s.pids 0*BM+i.1.val<M-1 then s.pids 0*BM+i.1.val else M-1)*N*K+
      (if i.2.1.val<N-1 then i.2.1.val else N-1)*K+
      (if s.pids 1<K-1 then s.pids 1 else K-1))<M*N*K := by
  simpa only [Address.offset,Address.safe,min_def_lt] using
    Address.masked_offset_bound (show 0<M by omega) (show 0<N by omega)
      (show 0<K by omega) (s.pids 0*BM+i.1.val) i.2.1.val (s.pids 1)

/-- Safety follows the actual masked FP32 load/store statements. Floating
primitive values, special values and the selected reductions are unrestricted. -/
theorem forward_source [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s : State A) (hc : 0<C) (candidate : Bool) (bounds : RegionName → Nat)
    (input : ∀ i, i<M*N*K → i<bounds X)
    (output : ∀ i, i<M*N*K → i<bounds O) :
    TraceSafe algebra bounds (if candidate then Source.forwardCommuted O X M N K BM C
      else Source.forward O X M N K BM C) s := by
  cases candidate <;>
  simp [TraceSafe,Source.forward,Source.forwardCommuted,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,
    TileShape.axisDim,hc,Nat.ne_of_gt hc,State.setReg,and_assoc,Nat.mul_comm BM]
  all_goals constructor
  all_goals
    intro i j hm hn hk
    first | apply input | apply output
    simpa only [ite_mul] using active_bound M N K BM C s (i,j,PUnit.unit) ⟨hm,hn,hk⟩

theorem backward_source [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) (candidate : Bool) (bounds : RegionName → Nat)
    (saved : ∀ i, i<M*N*K → i<bounds Y)
    (gradient : ∀ i, i<M*N*K → i<bounds G)
    (output : ∀ i, i<M*N*K → i<bounds O) :
    TraceSafe algebra bounds (if candidate then Source.backwardCommuted Y G O M N K BM C
      else Source.backward Y G O M N K BM C) s := by
  cases candidate <;>
  simp [TraceSafe,Source.backward,Source.backwardCommuted,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,
    TileShape.axisDim,State.setReg,and_assoc,Nat.mul_comm BM]
  all_goals refine ⟨?_,?_,?_⟩
  all_goals
    intro i j hm hn hk
    first | apply saved | apply gradient | apply output
    simpa only [ite_mul] using active_bound M N K BM C s (i,j,PUnit.unit) ⟨hm,hn,hk⟩

end
end VeriTile.Bench.Repairs.LogSoftmax.KernelSafety
