import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyParts

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.BackwardNoninner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPBackwardNoninner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem accumulate_safe (A : Algebra α) (base s : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Values α .real [B,C])
    (hs : Ready base row col N K B C cur v s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N*K → k<bounds P) (hg : ∀ k,k<rows*N*K → k<bounds G) :
    traceSafeList A bounds (rangeBody (code P G Y rows N K B C) 3)
      (s.setReg "_" .int [] (fun _ => (cur : Int))) := by
  rcases hs with ⟨hpid,hm,hn,ho,hv,hmem⟩
  simp [rangeBody,code,branch,Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hpid,hm,hn,ho,hv,hmem,offset]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  constructor
  · intro i j hn hk
    exact hp _ (RealAddress.non_inner_bound hrow hn hk)
  · intro i j hn hk
    exact hg _ (RealAddress.non_inner_bound hrow hn hk)

theorem emit_safe (A : Algebra α) (base s : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Values α .real [C])
    (hs : WriteReady base row col N K B C cur v s) (hrow : row<rows) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N*K → k<bounds P) (hg : ∀ k,k<rows*N*K → k<bounds G)
    (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    traceSafeList A bounds (rangeBody (code P G Y rows N K B C) 7)
      (s.setReg "_" .int [] (fun _ => (cur : Int))) := by
  rcases hs with ⟨hpid,hm,hn,ho,hv,hmem⟩
  simp [rangeBody,code,branch,Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hpid,hm,hn,ho,hv,hmem,offset]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  refine ⟨?_,?_,?_⟩
  · intro i j hn hk
    exact hp _ (RealAddress.non_inner_bound hrow hn hk)
  · intro i j hn hk
    exact hg _ (RealAddress.non_inner_bound hrow hn hk)
  · intro i j hn hk
    exact hy _ (RealAddress.non_inner_bound hrow hn hk)


theorem initial_safety (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) (bounds : RegionName → Nat) :
    TraceSafe A bounds (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false) s ↔
      rangeIntSafe A bounds "_" (0 : Int) (N : Int) B (rangeBody (code P G Y rows N K B C) 3)
        (initial A s N K B C) ∧
      ∀ mid, rangeInt A "_" (0 : Int) (N : Int) B (rangeBody (code P G Y rows N K B C) 3)
        (initial A s N K B C)=some mid →
        traceSafeList A bounds ((code P G Y rows N K B C).drop 4) mid := by
  let kernel := Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false
  let before := kernel.surfaceBody.take 3
  let yes := match kernel.surfaceBody[3]? with
    | some (.ifThenElse _ yes _) => yes | _ => []
  let ready := ((s.setReg "pid_m" .int [] (fun _ => (s.pids 0 : Int))).setReg
    "pid_k" .int [] (fun _ => (s.pids 1 : Int))).setReg "offsets_k" .int [C]
      (fun i => ((s.pids 1*C+i.1.val : Nat) : Int))
  have parts : kernel.surfaceBody = before++[.ifThenElse (.constBool Bool.false) yes
      (code P G Y rows N K B C)] := rfl
  have prefixFacts : traceSafeList A bounds before s ∧ run A before s=some ready := by
    constructor <;>
      simp [before,kernel,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalOp_unfold,Option.bind,
        numeric,bop,State.setReg,ready]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals rfl
  change traceSafeList A bounds kernel.surfaceBody s ↔ _
  rw [parts,prefix_iff A bounds before _ s ready prefixFacts.1 prefixFacts.2,false_branch_iff]
  let init := (code P G Y rows N K B C).take 3
  have splitBody : code P G Y rows N K B C = init++(code P G Y rows N K B C).drop 3 := rfl
  have initialized : traceSafeList A bounds init ready ∧
      run A init ready=some (initial A s N K B C) := by
    constructor <;>
      simp [init,code,branch,Source.softmax_backward_kernel_non_inner_fp32,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,
        VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
        ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
        safeCompute_unfold,safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
        numeric,bop,State.setReg,ready,initial,offset,ComputeDType.eraseDType,literal]
    all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    all_goals rfl
  conv_lhs => rw [splitBody]
  rw [prefix_iff A bounds init _ ready (initial A s N K B C) initialized.1 initialized.2]
  apply signed_loop_tail_iff A bounds "_" _ _ _ _ _ _ (0 : Int) (N : Int) B <;>
    simp [safeOp_unfold,evalOp_unfold]

theorem after_safety (A : Algebra α) (base mid : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Values α .real [B,C])
    (hs : Ready base row col N K B C cur v mid) (bounds : RegionName → Nat) :
    traceSafeList A bounds ((code P G Y rows N K B C).drop 4) mid ↔
      rangeIntSafe A bounds "_" (0 : Int) (N : Int) B (rangeBody (code P G Y rows N K B C) 7)
        (writeInitial A mid row col N K B C v) := by
  rcases hs with ⟨hpid,hm,hn,ho,hv,hmem⟩
  simp [code,branch,rangeBody,Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,ofFloat,toFloat,
    TileShape.axisDim,State.setReg,and_assoc,hpid,hm,hn,ho,hv,hmem,writeInitial,reduced,offset]
  all_goals delta offset bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  all_goals first | rfl | (split <;> simp)

theorem source (A : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) (hrow : s.pids 0<rows) (bounds : RegionName → Nat)
    (hp : ∀ k,k<rows*N*K → k<bounds P) (hg : ∀ k,k<rows*N*K → k<bounds G)
    (hy : ∀ k,k<rows*N*K → k<bounds Y) :
    TraceSafe A bounds (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false) s := by
  let f := accumulate A s P G (s.pids 0) (s.pids 1) rows N K B C
  have safe₁ := range_safe_int A bounds "_" (rangeBody (code P G Y rows N K B C) 3)
    (fun cur v st => Ready s (s.pids 0) (s.pids 1) N K B C cur v st) f N B
    (fun cur v st _ hs => accumulate_safe A s st P G Y (s.pids 0) (s.pids 1) rows N K B C cur v hs hrow bounds hp hg)
    (fun cur v st _ hs => accumulate_step A s st P G Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 (fun _ => literal A 0) (initial A s N K B C) (initial_ready A s N K B C)
  obtain ⟨mid,ran,ready⟩ := range_model_int A "_" (rangeBody (code P G Y rows N K B C) 3)
    f (fun cur v st => Ready s (s.pids 0) (s.pids 1) N K B C cur v st) N B
    (fun cur v st _ hs => accumulate_step A s st P G Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 (fun _ => literal A 0) (initial A s N K B C) (initial_ready A s N K B C)
  obtain ⟨_,outReady⟩ := after_accumulate A s mid P G Y (s.pids 0) (s.pids 1) rows N K B C
    (walk f N B 0 (fun _ => literal A 0)).1 (walk f N B 0 (fun _ => literal A 0)).2 ready
  apply (initial_safety A s P G Y rows N K B C bounds).mpr
  refine ⟨safe₁,?_⟩
  intro actual ha
  have : actual=mid := Option.some.inj (ha.symm.trans ran)
  subst actual
  apply (after_safety A s mid P G Y (s.pids 0) (s.pids 1) rows N K B C _ _ ready bounds).mpr
  exact range_safe_int A bounds "_" (rangeBody (code P G Y rows N K B C) 7)
    (fun cur base st => WriteReady base (s.pids 0) (s.pids 1) N K B C cur _ st)
    (emit A P G Y (s.pids 0) (s.pids 1) rows N K B C _) N B
    (fun cur base st _ hs => emit_safe A base st P G Y (s.pids 0) (s.pids 1) rows N K B C cur _ hs hrow bounds hp hg hy)
    (fun cur base st _ hs => emit_step A base st P G Y (s.pids 0) (s.pids 1) rows N K B C cur _ hs)
    0 s _ outReady

#axiomsClean source

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.BackwardNoninner
