import bench.repairs.tritonbench_g.kldiv_triton.Proofs.BackwardLoop
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KldivTriton.KernelSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 200000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

open Backward in
theorem backward_block {α : Type} [Inhabited α] (c : Tile.Invocation α)
    (ready : Ready c) (bounds : RegionName → Nat)
    (target : ∀ j : Fin c.B, c.current+j.val<c.N →
      c.tBase+(c.current+j.val)<bounds "target_ptr")
    (output : ∀ j : Fin c.B, c.current+j.val<c.N →
      c.oBase+(c.current+j.val)<bounds "input_ptr") :
    traceSafeList c.m.alg bounds (body c.so c.st c.N c.B c.flag) c.initial := by
  change traceSafeList c.m.alg bounds
    [bodyCommand c 0,bodyCommand c 1,bodyCommand c 2,bodyCommand c 3,bodyCommand c 4] (b0 c)
  simp only [traceSafeList_unfold,block0 c ready,block1 c ready,block2 c ready,
    block3 c ready,block4 c ready,and_true]
  refine ⟨?_,?_,?_,?_,?_⟩
  all_goals simp only [bodyCommand,body,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,numeric,numericLt,bop,ofFloat,toFloat,
    b0,b1,b2,b3,b4,State.setReg,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,ready.t,ready.o,
    ←Int.natCast_add,Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
  case refine_3 => exact fun i hi => target i.1 hi
  case refine_4 => split <;> split <;> trivial
  case refine_5 => exact fun i hi => output i.1 hi

open Backward in
theorem backward_prefix {α : Type} [Inhabited α] (c : Tile.Invocation α)
    (bounds : RegionName → Nat) :
    traceSafeList c.m.alg bounds
      ((backwardSource c.so c.st c.N c.B c.flag).surfaceBody.take 5) c.initial := by
  change traceSafeList c.m.alg bounds
    [command c 0,command c 1,command c 2,command c 3,command c 4] (a0 c)
  simp only [traceSafeList_unfold,initialize0,initialize1,initialize2,initialize3,
    initialize4,and_true]
  simp only [command,backwardSource,backwardTemplate,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,and_true]

open Backward in
theorem backward_range {α : Type} [Inhabited α] (m : Tile.Model α)
    (so st N B tBase oBase : Nat) (flag : Bool) (positive : 0<B)
    (bounds : RegionName → Nat)
    (target : ∀ j, j<N → tBase+j<bounds "target_ptr")
    (output : ∀ j, j<N → oBase+j<bounds "input_ptr")
    (count k : Nat) (s : State α)
    (ends : k+count=VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds N B)
    (header : Header s tBase oBase) :
    rangeIntSafe m.alg bounds "i" ((k*B : Nat) : Int) (N : Int) B
      (body so st N B flag) s := by
  induction count generalizing k s with
  | zero =>
      have hk : k=VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds N B := by omega
      have covered : N≤k*B := by
        rw [hk]
        exact VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds_cover N B positive
      rw [rangeIntSafe_unfold]
      simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,not_lt.mpr covered]
  | succ count ih =>
      have active : k*B<N :=
        (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.lt_rounds_iff N B k positive).mp (by omega)
      let c := invocation m (s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int)))
        so st N B tBase oBase (k*B) flag
      have ready : Ready c := by constructor <;> simp [c,invocation,header.t,header.o]
      have stepRun := block c ready
      dsimp only [c,invocation] at stepRun
      have next : ((k*B : Nat) : Int)+(B:Int)=(((k+1)*B:Nat):Int) := by push_cast;ring
      rw [rangeIntSafe_unfold]
      simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,active,if_true]
      refine ⟨?_,?_⟩
      · exact backward_block c ready bounds (fun j hj => target _ hj) (fun j hj => output _ hj)
      · rw [stepRun]
        simp only [next]
        exact ih (k+1) _ (by omega) (tick_header m s so st N B tBase oBase k flag header)

open Backward in
/-- Every actually executed backward load/store is within its declared region.
The proof follows the original signed loop and both original branches. -/
theorem backward_source {α : Type} [Inhabited α] (m : Tile.Model α)
    (so st N B : Nat) (flag : Bool) (positive : 0<B)
    (bounds : RegionName → Nat) (s : State α)
    (target : ∀ j, j<N → s.pids 0*st+j<bounds "target_ptr")
    (output : ∀ j, j<N → s.pids 0*so+j<bounds "input_ptr") :
    FP.Structural.TraceSafe m.alg bounds (backwardSource so st N B flag) s := by
  let c := invocation m s so st N B 0 0 0 flag
  let start := a5 c
  have initialized := source_initialize c
  unfold FP.Structural.TraceSafe
  rw [←List.take_append_drop 5 (backwardSource so st N B flag).surfaceBody,
    traceSafeList_append]
  refine ⟨backward_prefix c bounds,?_⟩
  intro final ran
  have same : final=start := Option.some.inj (ran.symm.trans initialized)
  subst final
  have header : Header start (s.pids 0*st) (s.pids 0*so) := by
    constructor <;> simp [start,c,invocation,a5,a4,a3,a2,a1,a0]
  have loop := backward_range m so st N B (s.pids 0*st) (s.pids 0*so) flag positive
    bounds target output (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds N B)
    0 start (by omega) header
  simp only [backwardSource,backwardTemplate,ComputeKernel.surfaceBody,List.drop]
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,
      Option.pure_def,Option.bind,true_and]
    simpa only [Nat.zero_mul,Nat.cast_zero] using loop
  · intro final _
    exact traceSafeList_nil _ _ _

open Tile in
theorem forward_block {α : Type} [Inhabited α] (c : Invocation α)
    (ready : Ready c) (bounds : RegionName → Nat)
    (prediction : ∀ j : Fin c.B, c.current+j.val<c.N →
      c.yBase+(c.current+j.val)<bounds "y_ptr")
    (target : ∀ j : Fin c.B, c.current+j.val<c.N →
      c.tBase+(c.current+j.val)<bounds "gt_ptr")
    (pointwise : c.reduction=0 → ∀ j : Fin c.B, c.current+j.val<c.N →
      c.oBase+(c.current+j.val)<bounds "loss_ptr")
    (reduced : c.reduction≠0 → c.oBase<bounds "loss_ptr") :
    traceSafeList c.m.alg bounds (forwardBody c.sy c.st c.so c.N c.B c.flag c.reduction) c.initial := by
  change traceSafeList c.m.alg bounds
    [bodyCommand c 0,bodyCommand c 1,bodyCommand c 2,bodyCommand c 3,
      bodyCommand c 4,bodyCommand c 5,bodyCommand c 6] (b0 c)
  simp only [traceSafeList_unfold,block0 c ready,block1 c ready,block2 c ready,
    block3 c ready,block4 c ready,block5 c ready,block6 c ready,and_true]
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  all_goals simp only [bodyCommand,forwardBody,forwardSource,forwardTemplate,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,numeric,numericLt,numericEq,FP.Structural.bop,ofFloat,toFloat,
    b0,b1,b2,b3,b4,b5,b6,f1,f2,f3,f4,f5,f6,State.setReg,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
    Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,ready.y,ready.t,ready.o,
    ←Int.natCast_add,Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
  case refine_3 => exact fun i hi => prediction i.1 hi
  case refine_4 => exact fun i hi => target i.1 hi
  case refine_5 =>
    split <;> repeat' first | trivial | split
  case refine_7 =>
    cases hf : c.flag <;>
      simp only [hf,Bool.false_eq_true,ite_true,ite_false,
        Option.bind_eq_bind,Option.pure_def,Option.bind,Option.some.injEq,
        forall_eq,forall_eq',State.setReg,ready.o,
        ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
        decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        ←Int.natCast_add,Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
    all_goals by_cases hr : c.reduction=0
    all_goals simp only [hr,ite_true,ite_false,exists_eq_left',decide_eq_true_eq]
    all_goals first
      | (refine ⟨fun i hi => pointwise hr i.1 hi,?_⟩; split <;> trivial)
      | (simp only [step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
          Option.bind_eq_bind,Option.pure_def,Option.bind,State.setReg,
          ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
          ite_true,ite_false,dite_true,dite_false,Option.some.injEq,forall_eq,forall_eq',
          ready.o,ComputeDType.eraseDType,ofFloat,toFloat]
         refine ⟨fun _ _ => reduced hr,?_⟩
         repeat' first | trivial | split)

open Tile in
theorem forward_prefix {α : Type} [Inhabited α] (c : Invocation α)
    (bounds : RegionName → Nat) :
    traceSafeList c.m.alg bounds
      ((forwardSource c.sy c.st c.so c.N c.B c.flag c.reduction).surfaceBody.take 5) c.initial := by
  change traceSafeList c.m.alg bounds
    [command c 0,command c 1,command c 2,command c 3,command c 4] (a0 c)
  simp only [traceSafeList_unfold,initialize0,initialize1,initialize2,initialize3,
    initialize4,and_true]
  simp only [command,forwardSource,forwardTemplate,ComputeKernel.surfaceBody,
    List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,and_true]

open Tile in
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath in
theorem forward_range {α : Type} [Inhabited α] (m : Model α)
    (sy st so yBase tBase base N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (bounds : RegionName → Nat)
    (prediction : ∀ j, j<N → yBase+j<bounds "y_ptr")
    (target : ∀ j, j<N → tBase+j<bounds "gt_ptr")
    (output : ∀ j, j<outputCount N B reduction → base+j<bounds "loss_ptr")
    (count k : Nat) (s : State α) (ends : k+count=rounds N B)
    (header : Header s yBase tBase (outputBase base k reduction) B) :
    rangeIntSafe m.alg bounds "i" ((k*B : Nat) : Int) (N : Int) B
      (forwardBody sy st so N B flag reduction) s := by
  induction count generalizing k s with
  | zero =>
      have hk : k=rounds N B := by omega
      have covered : N≤k*B := by rw [hk];exact rounds_cover N B positive
      rw [rangeIntSafe_unfold]
      simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,not_lt.mpr covered]
  | succ count ih =>
      have active : k*B<N := (lt_rounds_iff N B k positive).mp (by omega)
      let c : Invocation α := ⟨m,sy,st,so,N,B,flag,reduction,yBase,tBase,
        outputBase base k reduction,k*B,s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int))⟩
      have ready : Ready c :=
        set_i_ready m s sy st so yBase tBase (outputBase base k reduction) N B (k*B) flag reduction header
      have stepRun := forward_step c ready
      have next : ((k*B : Nat) : Int)+(B:Int)=(((k+1)*B:Nat):Int) := by push_cast;ring
      rw [rangeIntSafe_unfold]
      simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,active,if_true]
      refine ⟨?_,?_⟩
      · apply forward_block c ready bounds (fun j hj => prediction _ hj) (fun j hj => target _ hj)
        · intro mode j hj
          have hr : reduction=0 := mode
          simpa only [c,outputBase,hr,if_true,Nat.add_zero] using
            output (k*B+j.val) (by simpa only [outputCount,hr,if_true] using hj)
        · intro mode
          have hr : reduction≠0 := mode
          simpa only [c,outputBase,hr,if_false] using
            output k (by simp only [outputCount,hr,if_false];omega)
      · rw [stepRun]
        simp only [next]
        exact ih (k+1) _ (by omega) (tick_header m s yBase tBase base N B k flag reduction header)

open Tile in
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath in
/-- Complete forward source safety, including either pointwise masked stores or
one unmasked scalar store for each actual reduction chunk. -/
theorem forward_source {α : Type} [Inhabited α] (m : Model α)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (bounds : RegionName → Nat) (s : State α)
    (prediction : ∀ j, j<N → s.pids 0*sy+j<bounds "y_ptr")
    (target : ∀ j, j<N → s.pids 0*st+j<bounds "gt_ptr")
    (output : ∀ j, j<outputCount N B reduction → s.pids 0*so+j<bounds "loss_ptr") :
    FP.Structural.TraceSafe m.alg bounds (forwardSource sy st so N B flag reduction) s := by
  let c : Invocation α := ⟨m,sy,st,so,N,B,flag,reduction,0,0,0,0,s⟩
  let start := forwardInitial s sy st so B
  have initialized := forward_initialize m s sy st so N B flag reduction
  unfold FP.Structural.TraceSafe
  rw [source_decomposition,traceSafeList_append]
  refine ⟨forward_prefix c bounds,?_⟩
  intro final ran
  have same : final=start := Option.some.inj (ran.symm.trans initialized)
  subst final
  have header : Header start (s.pids 0*sy) (s.pids 0*st)
      (outputBase (s.pids 0*so) 0 reduction) B := by
    simpa [start,outputBase] using initial_header s sy st so B
  have loop := forward_range m sy st so (s.pids 0*sy) (s.pids 0*st) (s.pids 0*so)
    N B flag reduction positive bounds prediction target output (rounds N B) 0 start (by omega) header
  simp only [forwardSource,forwardTemplate,ComputeKernel.surfaceBody,List.drop]
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,
      Option.pure_def,Option.bind,true_and]
    simpa only [Nat.zero_mul,Nat.cast_zero] using loop
  · intro final _
    exact traceSafeList_nil _ _ _

#axiomsClean backward_block
#axiomsClean backward_prefix
#axiomsClean backward_range
#axiomsClean backward_source
#axiomsClean forward_block
#axiomsClean forward_prefix
#axiomsClean forward_range
#axiomsClean forward_source
end
end VeriTile.Bench.Repairs.KldivTriton.KernelSafety
