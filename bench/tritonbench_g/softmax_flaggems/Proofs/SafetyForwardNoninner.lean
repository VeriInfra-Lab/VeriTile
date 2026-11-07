import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyParts

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
      (s.setReg "start_n" .int [] (fun _ => (cur : Int))) := by
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hr,hk,hm,hz,hmem]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  intro i j hn hk
  exact hx _ (RealAddress.non_inner_bound hrow hn hk)

theorem emit_safe (A : Algebra α) (base s : State α) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (mx z : Values α .real [C])
    (hcur : cur≤previous N B)
    (hs : WriteReady base row col N B C mx z s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    traceSafeList A bounds (rangeBody (code Y X rows N K B C) 10)
      (s.setReg "start_n" .int [] (fun _ => (cur : Int))) := by
  have subCast : (previous N B : Int)-(cur : Int) = ((previous N B-cur : Nat) : Int) :=
    (Int.ofNat_sub hcur).symm
  rcases hs with ⟨hr,hk,hm,hz,hprev,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hr,hk,hm,hz,hprev,hmem]
  simp only [subCast,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  constructor
  · intro i j hn hk
    exact hx _ (RealAddress.non_inner_bound hrow hn hk)
  · intro i j hn hk
    exact hy _ (RealAddress.non_inner_bound hrow hn hk)

theorem initial_safety (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.forward_noninner Y X rows N K B C Bool.false) s ↔
      rangeIntSafe A bounds "start_n" (0 : Int) (N : Int) B (rangeBody (code Y X rows N K B C) 2)
        (initial A s B C) ∧
      ∀ mid, rangeInt A "start_n" (0 : Int) (N : Int) B (rangeBody (code Y X rows N K B C) 2)
        (initial A s B C)=some mid →
        traceSafeList A bounds ((code Y X rows N K B C).drop 3) mid := by
  let kernel := Source.forward_noninner Y X rows N K B C Bool.false
  let before := kernel.surfaceBody.take 3
  let yes := match kernel.surfaceBody[3]? with
    | some (.ifThenElse _ yes _) => yes | _ => []
  let ready := ((s.setReg "pid_k" .int [] (fun _ => (s.pids 1 : Int))).setReg
    "pid_m" .int [] (fun _ => (s.pids 0 : Int))).setReg "k_offsets" .int [C]
      (fun i => ((s.pids 1*C+i.1.val : Nat) : Int))
  have parts : kernel.surfaceBody = before++[.ifThenElse (.constBool Bool.false) yes
      (code Y X rows N K B C)] := rfl
  have prefixFacts : traceSafeList A bounds before s ∧ run A before s=some ready := by
    constructor <;>
      simp [before,kernel,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalOp_unfold,bind,Option.bind,
        numeric,bop,State.setReg,ready]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals rfl
  change traceSafeList A bounds kernel.surfaceBody s ↔ _
  rw [parts,prefix_iff A bounds before _ s ready prefixFacts.1 prefixFacts.2,false_branch_iff]
  let init := (code Y X rows N K B C).take 2
  have splitBody : code Y X rows N K B C = init++(code Y X rows N K B C).drop 2 := rfl
  have initialized : traceSafeList A bounds init ready ∧
      run A init ready=some (initial A s B C) := by
    constructor <;>
      simp [init,code,branch,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,bind,Option.bind,
        numeric,bop,State.setReg,ready,initial,offset,ComputeDType.eraseDType,literal,FPConstants.decimal_zero]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp]
    all_goals try rw [helperValue]
    all_goals rfl
  conv_lhs => rw [splitBody]
  rw [prefix_iff A bounds init _ ready (initial A s B C) initialized.1 initialized.2]
  apply signed_loop_tail_iff A bounds "start_n" _ _ _ _ _ _ (0 : Int) (N : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold]

theorem after_safety (A : Algebra α) (base mid : State α) (X Y : RegionName)
    (row col rows N K B C : Nat) (hN : 0<N) (hB : 0<B) (v : Acc α B C)
    (hs : Ready base row col B C v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code Y X rows N K B C).drop 3) mid ↔
      rangeIntSafe A bounds "start_n" (0 : Int) (N : Int) B (rangeBody (code Y X rows N K B C) 10)
        (writeInitial A mid N B C v) := by
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
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  let init := ((code Y X rows N K B C).drop 3).take 7
  have parts : (code Y X rows N K B C).drop 3 =
      init++(code Y X rows N K B C).drop 10 := rfl
  have initialized : traceSafeList A bounds init mid ∧
      run A init mid=some (writeInitial A mid N B C v) := by
    constructor <;>
      simp [init,code,branch,Source.forward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,bind,Option.bind,
        numeric,numericLt,natLt,bop,State.setReg,hm,hz,hmem,writeInitial,maxReduced,sumReduced,
        previous,integralFloorDiv,hB,FPReference.mul,FPReference.sub,FPReference.exp,ComputeDType.eraseDType,TileShape.axisDim]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [Option.bind,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
    all_goals
      have eta (i : TileIndex [B,C]) : (i.1,i.2.1,PUnit.unit)=i := by
        rcases i with ⟨a,b,u⟩; cases u; rfl
      try simp only [Function.comp_apply,eta]
    all_goals try simp only [if_neg hn,hp]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp,option_true,bind,Option.bind_some,Option.bind_none]
    all_goals try dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [if_neg hn,hp]
    all_goals try rw [helperValue]
    all_goals rfl
  rw [parts,prefix_iff A bounds init _ mid (writeInitial A mid N B C v) initialized.1 initialized.2]
  change traceSafeList A bounds [.forRangeInt "start_n" (.castNatToInt (.constNat 0))
    (.constInt (N : Int)) (.constNat B) (rangeBody (code Y X rows N K B C) 10)]
      (writeInitial A mid N B C v) ↔ _
  rw [singleton_iff]
  apply signed_loop_only_iff A bounds "start_n" _ _ _ _ _ (0 : Int) (N : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold]

theorem source (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (hN : 0<N) (hB : 0<B) (hrow : s.pids 0<rows) (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N*K → k<bounds X) (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    TraceSafe A bounds (Source.forward_noninner Y X rows N K B C Bool.false) s := by
  let f := accumulate A s X (s.pids 0) (s.pids 1) N K B C
  let seed : Acc α B C := ((fun _ => literal A (-340282346638528859811704183484516925440.0)),fun _ => literal A 0)
  have safe₁ := range_safe_int A bounds "start_n" (rangeBody (code Y X rows N K B C) 2)
    (fun _ v st => Ready s (s.pids 0) (s.pids 1) B C v st) f N B
    (fun cur v st _ hs => accumulate_safe A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs hrow bounds hx)
    (fun cur v st _ hs => accumulate_step A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 seed (initial A s B C) (initial_ready A s B C)
  obtain ⟨mid,ran,ready⟩ := range_model_int A "start_n" (rangeBody (code Y X rows N K B C) 2)
    f (fun _ v st => Ready s (s.pids 0) (s.pids 1) B C v st) N B
    (fun cur v st _ hs => accumulate_step A s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 seed (initial A s B C) (initial_ready A s B C)
  obtain ⟨_,outReady⟩ := after_accumulate A s mid X Y (s.pids 0) (s.pids 1) rows N K B C hN hB
    (walk f N B 0 seed).2 ready
  apply (initial_safety A s X Y rows N K B C bounds).mpr
  refine ⟨safe₁,?_⟩
  intro actual ha
  have : actual=mid := Option.some.inj (ha.symm.trans ran)
  subst actual
  apply (after_safety A s mid X Y (s.pids 0) (s.pids 1) rows N K B C hN hB _ ready bounds).mpr
  exact range_safe_int_aligned A bounds "start_n" (rangeBody (code Y X rows N K B C) 10)
    (fun _ base st => WriteReady base (s.pids 0) (s.pids 1) N B C _ _ st)
    (emit A X Y (s.pids 0) (s.pids 1) N K B C _ _) N B
    (fun cur base st hc aligned hs => emit_safe A base st X Y (s.pids 0) (s.pids 1) rows N K B C cur _ _
      (SignedAddress.tile_le_previous N B cur hN hB hc aligned) hs hrow bounds hx hy)
    (fun cur base st hc aligned hs => emit_step A base st X Y (s.pids 0) (s.pids 1) rows N K B C cur _ _
      (SignedAddress.tile_le_previous N B cur hN hB hc aligned) hs)
    0 s _ (dvd_zero B) outReady

#axiomsClean source
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardNoninner
