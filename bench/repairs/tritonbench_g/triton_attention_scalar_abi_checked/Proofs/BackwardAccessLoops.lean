import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardAccessBody
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardAccessStart
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.BackwardAccessFinish
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000

theorem backward_inner_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block n : Nat) (mem : Memory α)
    (positive : 0<p.blockM)
    (inv : BackwardInnerInvariant p initial block (block+n) mem
      (backwardInnerRounds P p initial mem block n) current) (bounds : RegionName→Nat)
    (elements : ∀ query,query<p.blocks → ∀ i : BLane p,
      backwardOffset p initial query i<bounds p.q ∧
      backwardOffset p initial query i<bounds p.gradient ∧
      backwardOffset p initial query i<bounds p.dq)
    (rows : ∀ query,query<p.blocks → ∀ i : BRow p,
      backwardRowOffset p initial query i<bounds p.maximum ∧
      backwardRowOffset p initial query i<bounds p.delta) :
    Safety.rangeSafe (backend rest P) bounds "start_m" ((block+n)*p.blockM)
      (p.blocks*p.blockM) p.blockM (backwardInnerBody p) current := by
  apply Safety.rangeSafe_of_invariant (backend rest P) bounds "start_m" (p.blocks*p.blockM) p.blockM
    (backwardInnerBody p) (fun offset s=>∃ j,offset=(block+j)*p.blockM ∧
      BackwardInnerInvariant p initial block (block+j) mem
        (backwardInnerRounds P p initial mem block j) s) _ _ _ _ ⟨n,rfl,inv⟩
  · intro offset active s invariant
    obtain ⟨j,rfl,h⟩ := invariant
    have live : block+j<p.blocks := (Nat.mul_lt_mul_right positive).mp active
    exact backward_body_safe rest P p initial s block (block+j) mem _ h bounds
      (fun i=>(elements _ live i).1) (fun i=>(rows _ live i).1)
      (fun i=>(elements _ live i).2.1) (fun i=>(rows _ live i).2)
      (fun i=>(elements _ live i).2.2)
  · intro offset active s invariant t ran
    obtain ⟨j,rfl,h⟩ := invariant
    rw [backwardBodyExecution rest P p initial s block (block+j) mem _ h] at ran
    cases ran
    refine ⟨j+1,by simp [Nat.add_mul,Nat.add_assoc],?_⟩
    simpa only [Nat.add_assoc] using backwardBodyInvariant P p initial s block (block+j) mem _ h

theorem backward_outer_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block : Nat) (mem : Memory α)
    (live : block<p.blocks) (positive : 0<p.blockM)
    (inv : BackwardEntryInvariant p initial block mem current) (bounds : RegionName→Nat)
    (elements : ∀ query,query<p.blocks → ∀ i : BLane p,
      backwardOffset p initial query i<bounds p.q ∧
      backwardOffset p initial query i<bounds p.gradient ∧
      backwardOffset p initial query i<bounds p.dq)
    (rows : ∀ query,query<p.blocks → ∀ i : BRow p,
      backwardRowOffset p initial query i<bounds p.maximum ∧
      backwardRowOffset p initial query i<bounds p.delta)
    (keys : ∀ i : BLane p,backwardOffset p initial block i<bounds p.k)
    (values : ∀ i : BLane p,backwardOffset p initial block i<bounds p.v)
    (dk : ∀ i : BLane p,backwardOffset p initial block i<bounds p.dk)
    (dv : ∀ i : BLane p,backwardOffset p initial block i<bounds p.dv) :
    Safety.traceSafeList (backend rest P) bounds (backwardOuterBody p)
      (current.setReg "start_n" .nat [] (fun _=>block)) := by
  let start := backwardOuterStartResult P p initial current block mem
  have startInv := backwardOuterStartInvariant P p initial current block mem inv
  have safeLoop := backward_inner_safe rest P p initial start block 0 mem positive startInv bounds elements rows
  obtain ⟨middle,loopRan,middleInv⟩ := backwardInnerExecution rest P p initial start block 0 mem (by omega) positive startInv
  have loopRun : run (backend rest P)
      [.forRangeDyn "start_m" (.ref .nat [] "lo")
        (.mul .nat .nil (.constNat p.blocks) (.constNat p.blockM))
        (.constNat p.blockM) (backwardInnerBody p)] start=some middle := by
    simpa [run,step,evalOp_unfold,numeric,startInv.lo,bop,
      Broadcast.leftIndex,Broadcast.rightIndex] using loopRan
  rw [backwardInnerDecomposition,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨backward_start_safe rest P p initial current block mem inv bounds keys values,?_⟩,?_⟩
  · intro next ran
    rw [backwardOuterStartExecution rest P p initial current block mem inv] at ran
    cases ran
    simpa [AccessSafety.singleton_safe,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,startInv.lo,Broadcast.leftIndex,Broadcast.rightIndex] using safeLoop
  · intro next ran
    rw [run_append,backwardOuterStartExecution rest P p initial current block mem inv] at ran
    simp only [Option.bind_some] at ran
    have equal : middle=next := Option.some.inj (loopRun.symm.trans ran)
    subst next
    exact backward_finish_safe rest P p initial middle block _ middleInv.toBackwardOuterInvariant
      middleInv.dv middleInv.dk middleInv.lo middleInv.qPtr middleInv.doPtr middleInv.dqPtr bounds dv dk

theorem backward_outer_loop_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block : Nat) (positive : 0<p.blockM)
    (inv : BackwardEntryInvariant p initial block (backwardRounds P p initial block) current)
    (bounds : RegionName→Nat)
    (elements : ∀ query,query<p.blocks → ∀ i : BLane p,
      backwardOffset p initial query i<bounds p.q ∧
      backwardOffset p initial query i<bounds p.gradient ∧
      backwardOffset p initial query i<bounds p.dq)
    (rows : ∀ query,query<p.blocks → ∀ i : BRow p,
      backwardRowOffset p initial query i<bounds p.maximum ∧
      backwardRowOffset p initial query i<bounds p.delta)
    (outputs : ∀ query,query<p.blocks → ∀ i : BLane p,
      backwardOffset p initial query i<bounds p.k ∧
      backwardOffset p initial query i<bounds p.v ∧
      backwardOffset p initial query i<bounds p.dk ∧
      backwardOffset p initial query i<bounds p.dv) :
    Safety.rangeSafe (backend rest P) bounds "start_n" block p.blocks 1 (backwardOuterBody p) current := by
  apply Safety.rangeSafe_of_invariant (backend rest P) bounds "start_n" p.blocks 1
    (backwardOuterBody p) (fun j s=>BackwardEntryInvariant p initial j (backwardRounds P p initial j) s)
    _ _ _ _ inv
  · intro j live s h
    exact backward_outer_safe rest P p initial s j _ live positive h bounds elements rows
      (fun i=>(outputs _ live i).1) (fun i=>(outputs _ live i).2.1)
      (fun i=>(outputs _ live i).2.2.1) (fun i=>(outputs _ live i).2.2.2)
  · intro j live s h t ran
    obtain ⟨next,ranNext,invariant⟩ := backwardOuterExecution rest P p initial s j _ (by omega) positive h
    have equal : next=t := Option.some.inj (ranNext.symm.trans ran)
    subst t
    exact invariant

#axiomsClean backward_inner_safe
#axiomsClean backward_outer_safe
#axiomsClean backward_outer_loop_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
