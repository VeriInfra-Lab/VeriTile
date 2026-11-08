import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyForwardInner
import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyParts

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem initial_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (hN : 0<N) (hB : 0<B) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.forward_inner Y X rows N B Bool.false) s ↔
      rangeIntSafe A bounds "start_n" (0 : Int) (previous N B : Int) B (rangeBody (code Y X rows N B) 8)
        (initial A s X Y N B) ∧
      ∀ mid, rangeInt A "start_n" (0 : Int) (previous N B : Int) B (rangeBody (code Y X rows N B) 8)
        (initial A s X Y N B)=some mid →
        traceSafeList A bounds ((code Y X rows N B).drop 9) mid := by
  have hn : ¬ (N : Int)+((B : Int)-1)<0 := by omega
  have hp : ((N : Int)+((B : Int)-1))/(B : Int)*(B : Int)-(B : Int) = (previous N B : Int) := by
    have hsum : (N : Int)+((B : Int)-1) = (N : Int)+(B : Int)-1 := by omega
    rw [hsum]
    exact SignedAddress.previous_eq N B hN hB
  have helperValue : FP.Structural.bop (fun x y : Int=>x-y) .nil
      (FP.Structural.bop (fun x y : Int=>x*y) .nil
        (fun _ : TileIndex []=>((N : Int)+((B : Int)-1))/(B : Int))
        (fun _ : TileIndex []=>(B : Int)))
      (fun _ : TileIndex []=>(B : Int)) = (fun _ : TileIndex []=>(previous N B : Int)) := by
    funext i; cases i; exact hp
  have pointerValue (R : RegionName) (offset : Nat) :
      FP.Structural.bop (fun (a : RegionName × Nat) (b : Nat)=>(a.1,a.2+b)) .nil
        (fun _ : TileIndex []=>(R,0)) (fun _ : TileIndex []=>offset) =
      (fun _ : TileIndex []=>(R,offset)) := by
    funext i; cases i; simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  let kernel := Source.forward_inner Y X rows N B Bool.false
  let before := kernel.surfaceBody.take 1
  let yes := match kernel.surfaceBody[1]? with
    | some (.ifThenElse _ yes _) => yes | _ => []
  let ready := s.setReg "pid_m" .int [] (fun _ => (s.pids 0 : Int))
  have parts : kernel.surfaceBody = before++[.ifThenElse (.constBool Bool.false) yes
      (code Y X rows N B)] := rfl
  have prefixFacts : traceSafeList A bounds before s ∧ run A before s=some ready := by
    constructor <;>
      simp [before,kernel,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalOp_unfold,bind,Option.bind,
        numeric,bop,State.setReg,ready]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals rfl
  change traceSafeList A bounds kernel.surfaceBody s ↔ _
  rw [parts,prefix_iff A bounds before _ s ready prefixFacts.1 prefixFacts.2,false_branch_iff]
  let init := (code Y X rows N B).take 8
  have splitBody : code Y X rows N B = init++(code Y X rows N B).drop 8 := rfl
  have initialized : traceSafeList A bounds init ready ∧
      run A init ready=some (initial A s X Y N B) := by
    constructor <;>
      simp [init,code,branch,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,bind,Option.bind,
        numeric,numericLt,natLt,integralFloorDiv,bop,State.setReg,ready,initial,ComputeDType.eraseDType,literal,FPConstants.decimal_zero]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals try simp only [if_neg hn,hp]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp,option_true,bind,Option.bind_some,Option.bind_none]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp]
    all_goals try rw [helperValue]
    all_goals try rw [pointerValue,pointerValue]
    all_goals rfl
  conv_lhs => rw [splitBody]
  rw [prefix_iff A bounds init _ ready (initial A s X Y N B) initialized.1 initialized.2]
  apply signed_loop_tail_iff A bounds "start_n" _ _ _ _ _ _ (0 : Int) (previous N B : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold,initial,State.setReg]

theorem between_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (v : Acc α B) (hs : Ready base X Y row N B v mid)
    (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 9) mid ↔
      rangeIntSafe A bounds "start_n" (previous N B : Int) (N : Int) B (rangeBody (code Y X rows N B) 9) mid ∧
      ∀ t, rangeInt A "start_n" (previous N B : Int) (N : Int) B (rangeBody (code Y X rows N B) 9) mid=some t →
        traceSafeList A bounds ((code Y X rows N B).drop 10) t := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  apply signed_loop_tail_iff A bounds "start_n" _ _ _ _ _ mid (previous N B : Int) (N : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold,hprev]

theorem after_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (hN : 0<N) (hB : 0<B) (v : Acc α B)
    (hs : Ready base X Y row N B v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 10) mid ↔
      rangeSafe A bounds "start_n" 0 B B (rangeBody (code Y X rows N B) 17)
        (writeInitial A mid N B v) ∧
      ∀ t, range A "start_n" 0 B B (rangeBody (code Y X rows N B) 17)
        (writeInitial A mid N B v)=some t →
        traceSafeList A bounds ((code Y X rows N B).drop 18) t := by
  have hn : ¬ (N : Int)+((B : Int)-1)<0 := by omega
  have hp : ((N : Int)+((B : Int)-1))/(B : Int)*(B : Int)-(B : Int) = (previous N B : Int) := by
    have hsum : (N : Int)+((B : Int)-1) = (N : Int)+(B : Int)-1 := by omega
    rw [hsum]
    exact SignedAddress.previous_eq N B hN hB
  have helperValue : FP.Structural.bop (fun x y : Int=>x-y) .nil
      (FP.Structural.bop (fun x y : Int=>x*y) .nil
        (fun _ : TileIndex []=>((N : Int)+((B : Int)-1))/(B : Int))
        (fun _ : TileIndex []=>(B : Int)))
      (fun _ : TileIndex []=>(B : Int)) = (fun _ : TileIndex []=>(previous N B : Int)) := by
    funext i; cases i; exact hp
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  let init := ((code Y X rows N B).drop 10).take 7
  have parts : (code Y X rows N B).drop 10 =
      init++(code Y X rows N B).drop 17 := rfl
  have initialized : traceSafeList A bounds init mid ∧
      run A init mid=some (writeInitial A mid N B v) := by
    constructor <;>
      simp [init,code,branch,Source.forward_inner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_inner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,bind,Option.bind,
        numeric,numericLt,natLt,bop,State.setReg,hm,hz,hmem,writeInitial,maxReduced,sumReduced,
        previous,integralFloorDiv,hB,FPReference.mul,FPReference.sub,FPReference.exp,
        ComputeDType.eraseDType,TileShape.axisDim]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [Option.bind,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
    all_goals
      have eta (i : TileIndex [B]) : (i.1,PUnit.unit)=i := by
        rcases i with ⟨a,u⟩; cases u; rfl
      have scalar_eta (f : Values α .real []) : (fun _ : TileIndex []=>f PUnit.unit)=f := by
        funext i; cases i; rfl
      try simp only [Function.comp_apply,eta,scalar_eta]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp,option_true,bind,Option.bind_some,Option.bind_none]
    all_goals try rw [helperValue]
    all_goals rfl
  rw [parts,prefix_iff A bounds init _ mid (writeInitial A mid N B v) initialized.1 initialized.2]
  change traceSafeList A bounds (.forRange "start_n" 0 B B (rangeBody (code Y X rows N B) 17)::
    (code Y X rows N B).drop 18) (writeInitial A mid N B v) ↔ _
  exact nat_loop_tail_iff A bounds "start_n" 0 B B _ _ _

theorem last_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N B).drop 18) s ↔
      rangeIntSafe A bounds "start_n" (B : Int) (N : Int) B (rangeBody (code Y X rows N B) 18) s := by
  change traceSafeList A bounds [.forRangeInt "start_n" (.castNatToInt (.constNat B))
    (.constInt (N : Int)) (.constNat B) (rangeBody (code Y X rows N B) 18)] s ↔ _
  rw [singleton_iff]
  apply signed_loop_only_iff A bounds "start_n" _ _ _ _ s (B : Int) (N : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold]

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
