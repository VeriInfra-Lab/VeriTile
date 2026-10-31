import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyBase

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardNoninner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardNoninner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem accumulate_safe (A : Algebra α) (base s : State α) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Acc α B C)
    (hs : Ready base row col B C v s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) :
    traceSafeList A bounds (rangeBody (code Y X rows N K B C) 2)
      (s.setReg "start_n" .nat [] (fun _ => cur)) := by
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hr,hk,hm,hz,hmem]
  intro i j hn hk
  exact hx _ (RealAddress.non_inner_bound hrow hn hk)

theorem emit_safe (A : Algebra α) (base s : State α) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (mx z : Values α .real [C])
    (hs : WriteReady base row col N B C mx z s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    traceSafeList A bounds (rangeBody (code Y X rows N K B C) 7)
      (s.setReg "start_n" .nat [] (fun _ => cur)) := by
  rcases hs with ⟨hr,hk,hm,hz,hprev,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hr,hk,hm,hz,hprev,hmem]
  constructor
  · intro i j hn hk
    exact hx _ (RealAddress.non_inner_bound hrow hn hk)
  · intro i j hn hk
    exact hy _ (RealAddress.non_inner_bound hrow hn hk)

theorem initial_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.forward_noninner Y X rows N K B C Bool.false) s ↔
      rangeSafe A bounds "start_n" 0 N B (rangeBody (code Y X rows N K B C) 2)
        (initial A s B C) ∧
      ∀ mid, range A "start_n" 0 N B (rangeBody (code Y X rows N K B C) 2)
        (initial A s B C)=some mid →
        traceSafeList A bounds ((code Y X rows N K B C).drop 3) mid := by
  simp [TraceSafe,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,code,branch,rangeBody,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,initial,literal]
  all_goals delta bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,FPConstants.decimal_zero]
  all_goals split <;> simp_all
  all_goals intros; split <;> trivial

theorem after_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row col rows N K B C : Nat) (hB : 0<B) (v : Acc α B C)
    (hs : Ready base row col B C v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N K B C).drop 3) mid ↔
      rangeSafe A bounds "start_n" 0 N B (rangeBody (code Y X rows N K B C) 7)
        (writeInitial A mid N B C v) := by
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hr,hk,hm,hz,hmem,writeInitial,maxReduced,sumReduced,
    previous,integralFloorDiv,hB,FPReference.mul,FPReference.sub,FPReference.exp]
  all_goals delta bop maxReduced
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  all_goals try simp only [Option.bind,hr,hk,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
  all_goals
    have eta (i : TileIndex [B,C]) : (i.1,i.2.1,PUnit.unit)=i := by
      rcases i with ⟨a,b,u⟩; cases u; rfl
    simp only [Function.comp_apply,eta]
  all_goals first | rfl | (split <;> simp)

theorem source (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (hB : 0<B) (hrow : s.pids 0<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    TraceSafe A bounds (Source.forward_noninner Y X rows N K B C Bool.false) s := by
  let f := accumulate A s X (s.pids 0) (s.pids 1) N K B C
  let seed : Acc α B C := ((fun _ => literal A (-340282346638528859811704183484516925440.0)),fun _ => literal A 0)
  have safe₁ := range_safe A bounds "start_n" (rangeBody (code Y X rows N K B C) 2)
    (fun _ v st => Ready s (s.pids 0) (s.pids 1) B C v st) f N B
    (fun cur v st _ hs => accumulate_safe A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs hrow bounds hx)
    (fun cur v st _ hs => accumulate_step A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 seed (initial A s B C) (initial_ready A s B C)
  obtain ⟨mid,ran,ready⟩ := range_model A "start_n" (rangeBody (code Y X rows N K B C) 2)
    f (fun _ v st => Ready s (s.pids 0) (s.pids 1) B C v st) N B
    (fun cur v st _ hs => accumulate_step A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 seed (initial A s B C) (initial_ready A s B C)
  obtain ⟨_,outReady⟩ := after_accumulate A s mid X Y (s.pids 0) (s.pids 1) rows N K B C hB
    (walk f N B 0 seed).2 ready
  apply (initial_safety A s X Y rows N K B C bounds).mpr
  refine ⟨safe₁,?_⟩
  intro actual ha
  have : actual=mid := Option.some.inj (ha.symm.trans ran)
  subst actual
  apply (after_safety A s mid X Y (s.pids 0) (s.pids 1) rows N K B C hB _ ready bounds).mpr
  exact range_safe A bounds "start_n" (rangeBody (code Y X rows N K B C) 7)
    (fun _ base st => WriteReady base (s.pids 0) (s.pids 1) N B C _ _ st)
    (emit A X Y (s.pids 0) (s.pids 1) N K B C _ _) N B
    (fun cur base st _ hs => emit_safe A base st X Y (s.pids 0) (s.pids 1) rows N K B C cur _ _ hs hrow bounds hx hy)
    (fun cur base st _ hs => emit_step A base st X Y (s.pids 0) (s.pids 1) rows N K B C cur _ _ hs)
    0 s _ outReady

#axiomsClean source
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardNoninner
