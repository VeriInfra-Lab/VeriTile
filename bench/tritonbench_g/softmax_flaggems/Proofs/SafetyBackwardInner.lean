import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.BackwardInner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPBackwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem accumulate_safe (A : Algebra α) (base s : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (v : Values α .real [C,B])
    (hs : Ready base row N B C cur v s) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N → k<bounds P) (hg : ∀ k,k<rows*N → k<bounds G) :
    traceSafeList A bounds (rangeBody (code P G Y rows N B C) 3)
      (s.setReg "unused_start_n" .nat [] (fun _ => cur)) := by
  rcases hs with ⟨hm,hn,ho,hv,hmem⟩
  simp [rangeBody,code,branch,Source.softmax_backward_kernel_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hm,hn,ho,hv,hmem,offset]
  constructor
  · intro i j hr hn
    exact hp _ (HostGrid.inner_bound hr hn)
  · intro i j hr hn
    exact hg _ (HostGrid.inner_bound hr hn)

theorem emit_safe (A : Algebra α) (base s : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (v : Values α .real [C])
    (hs : WriteReady base row N B C cur v s) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N → k<bounds P) (hg : ∀ k,k<rows*N → k<bounds G)
    (hy : ∀ k,k<rows*N → k<bounds Y) :
    traceSafeList A bounds (rangeBody (code P G Y rows N B C) 7)
      (s.setReg "unused_start_n" .nat [] (fun _ => cur)) := by
  rcases hs with ⟨hm,hn,ho,hv,hmem⟩
  simp [rangeBody,code,branch,Source.softmax_backward_kernel_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hm,hn,ho,hv,hmem,offset]
  refine ⟨?_,?_,?_⟩
  · intro i j hr hn
    exact hp _ (HostGrid.inner_bound hr hn)
  · intro i j hr hn
    exact hg _ (HostGrid.inner_bound hr hn)
  · intro i j hr hn
    exact hy _ (HostGrid.inner_bound hr hn)


theorem initial_safety (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.false) s ↔
      rangeSafe A bounds "unused_start_n" 0 N B (rangeBody (code P G Y rows N B C) 3)
        (initial A s N B C) ∧
      ∀ mid, range A "unused_start_n" 0 N B (rangeBody (code P G Y rows N B C) 3)
        (initial A s N B C)=some mid →
        traceSafeList A bounds ((code P G Y rows N B C).drop 4) mid := by
  simp [TraceSafe,Source.softmax_backward_kernel_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_inner,code,branch,rangeBody,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,initial,offset,literal]
  all_goals delta offset bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals split <;> simp_all
  all_goals intros; split <;> trivial

theorem after_safety (A : Algebra α) (base mid : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (v : Values α .real [C,B])
    (hs : Ready base row N B C cur v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code P G Y rows N B C).drop 4) mid ↔
      rangeSafe A bounds "unused_start_n" 0 N B (rangeBody (code P G Y rows N B C) 7)
        (writeInitial A mid row N B C v) := by
  rcases hs with ⟨hm,hn,ho,hv,hmem⟩
  simp [code,branch,rangeBody,Source.softmax_backward_kernel_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_inner,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hm,hn,ho,hv,hmem,writeInitial,reduced,offset]
  all_goals delta offset bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  all_goals first | rfl | (split <;> simp)

theorem source (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N → k<bounds P) (hg : ∀ k,k<rows*N → k<bounds G)
    (hy : ∀ k,k<rows*N → k<bounds Y) :
    TraceSafe A bounds (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.false) s := by
  let f := accumulate A s P G (s.pids 0) rows N B C
  have safe₁ := range_safe A bounds "unused_start_n" (rangeBody (code P G Y rows N B C) 3)
    (fun cur v st => Ready s (s.pids 0) N B C cur v st) f N B
    (fun cur v st _ hs => accumulate_safe A s st P G Y (s.pids 0) rows N B C cur v hs bounds hp hg)
    (fun cur v st _ hs => accumulate_step A s st P G Y (s.pids 0) rows N B C cur v hs)
    0 (fun _ => literal A 0) (initial A s N B C) (initial_ready A s N B C)
  obtain ⟨mid,ran,ready⟩ := range_model A "unused_start_n" (rangeBody (code P G Y rows N B C) 3)
    f (fun cur v st => Ready s (s.pids 0) N B C cur v st) N B
    (fun cur v st _ hs => accumulate_step A s st P G Y (s.pids 0) rows N B C cur v hs)
    0 (fun _ => literal A 0) (initial A s N B C) (initial_ready A s N B C)
  obtain ⟨_,outReady⟩ := after_accumulate A s mid P G Y (s.pids 0) rows N B C
    (walk f N B 0 (fun _ => literal A 0)).1 (walk f N B 0 (fun _ => literal A 0)).2 ready
  apply (initial_safety A s P G Y rows N B C bounds).mpr
  refine ⟨safe₁,?_⟩
  intro actual ha
  have : actual=mid := Option.some.inj (ha.symm.trans ran)
  subst actual
  apply (after_safety A s mid P G Y (s.pids 0) rows N B C _ _ ready bounds).mpr
  exact range_safe A bounds "unused_start_n" (rangeBody (code P G Y rows N B C) 7)
    (fun cur base st => WriteReady base (s.pids 0) N B C cur _ st)
    (emit A P G Y (s.pids 0) rows N B C _) N B
    (fun cur base st _ hs => emit_safe A base st P G Y (s.pids 0) rows N B C cur _ hs bounds hp hg hy)
    (fun cur base st _ hs => emit_step A base st P G Y (s.pids 0) rows N B C cur _ hs)
    0 s _ outReady

#axiomsClean source

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.BackwardInner
