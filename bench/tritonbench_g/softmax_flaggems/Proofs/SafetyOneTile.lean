import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_inner_one (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (hB : 0<B) (hr : s.pids 0<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N → k<bounds X) (hy : ∀ k,k<rows*N → k<bounds Y) :
    TraceSafe A bounds (Source.forward_inner Y X rows N B Bool.true) s := by
  simp [Source.forward_inner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,TraceSafe,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hB,Nat.ne_of_gt hB]
  constructor
  · intro i hi
    exact hx _ (HostGrid.inner_bound hr hi)
  · intro i hi
    exact hy _ (HostGrid.inner_bound hr hi)

theorem forward_noninner_one (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (hB : 0<B) (hr : s.pids 0<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    TraceSafe A bounds (Source.forward_noninner Y X rows N K B C Bool.true) s := by
  simp [Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,TraceSafe,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hB,Nat.ne_of_gt hB]
  constructor
  · intro i j hn hk
    exact hx _ (RealAddress.non_inner_bound hr hn hk)
  · intro i j hn hk
    exact hy _ (RealAddress.non_inner_bound hr hn hk)

theorem backward_inner_one (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N → k<bounds P) (hg : ∀ k,k<rows*N → k<bounds G)
    (hy : ∀ k,k<rows*N → k<bounds Y) :
    TraceSafe A bounds (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.true) s := by
  simp [Source.softmax_backward_kernel_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_inner,
    TraceSafe,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc]
  refine ⟨?_,?_,?_⟩
  · intro i j hr hn
    exact hp _ (HostGrid.inner_bound hr hn)
  · intro i j hr hn
    exact hg _ (HostGrid.inner_bound hr hn)
  · intro i j hr hn
    exact hy _ (HostGrid.inner_bound hr hn)

theorem backward_noninner_one (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) (hr : s.pids 0<rows) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N*K → k<bounds P) (hg : ∀ k,k<rows*N*K → k<bounds G)
    (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    TraceSafe A bounds (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.true) s := by
  simp [Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,
    TraceSafe,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc]
  refine ⟨?_,?_,?_⟩
  · intro i j hn hk
    exact hp _ (RealAddress.non_inner_bound hr hn hk)
  · intro i j hn hk
    exact hg _ (RealAddress.non_inner_bound hr hn hk)
  · intro i j hn hk
    exact hy _ (RealAddress.non_inner_bound hr hn hk)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
