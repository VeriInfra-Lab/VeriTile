import bench.tritonbench_g.matmul_triton_autotune.Proofs.Execution
import bench.tritonbench_g.matmul_triton_autotune.Proofs.SourceTotal

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton FP.Structural Whole
open FPValue
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 4000000

theorem written_mem_congr {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (p : Parameters) (s t : State α)
    (hm : s.mem=t.mem) (hp : s.pids=t.pids) :
    (written M D p .single s).mem=(written M D p .single t).mem := by
  have ha (k : Nat) : aValue M p .single s k=aValue M p .single t k := by
    funext i
    simp only [aValue,aOffset,aRow,row,pidM,firstM,FPValue.groupSize,hm,hp]
  have hb (k : Nat) : bValue M p .single s k=bValue M p .single t k := by
    funext i
    simp only [bValue,bOffset,bCol,col,pidN,firstM,FPValue.groupSize,hm,hp]
  have hu : update M D p .single s=update M D p .single t := by
    funext k a i
    simp only [update,ha,hb]
  simp only [written,masked_scatter_memory,outputValue,activation,accumulation,hu,
    active,cOffset,row,col,pidM,pidN,firstM,FPValue.groupSize,hm,hp]

/-- Both complete schedules have the same success/failure and full state.
The actual pasted programs differ only by an integer product commutation. -/
theorem programs_execution_equal {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat) (s : State α) :
    Whole.runPrograms M D r ids s=runPrograms M D r ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    change (FP.Structural.exec (backend M D) (FPValue.source r.parameters .single)
      (instanceState r pid s)).bind (Whole.runPrograms M D r rest)=
      (FP.Structural.exec (backend M D) (Source.kernel r.parameters)
        (instanceState r pid s)).bind (runPrograms M D r rest)
    rw [SourcePair.execution_equal]
    apply congrArg (fun f => (FP.Structural.exec (backend M D) (Source.kernel r.parameters)
      (instanceState r pid s)).bind f)
    funext t
    exact ih t

theorem original_grid_executes {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat) (s : State α) :
    ∃ t,Whole.runPrograms M D r ids s=some t := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons pid rest ih =>
    obtain ⟨next,he⟩:=FPValue.Total.executes M D r.parameters (instanceState r pid s)
    obtain ⟨final,hf⟩:=ih next
    exact ⟨final,by simpa only [Whole.runPrograms,kernel,he,Option.bind_some] using hf⟩

/-- Complete successful execution and equal memory are constructed from the
source, for arbitrary launch lists and initial states. -/
theorem programs_refinement {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat) (s : State α) :
    ∃ sf tf,Whole.runPrograms M D r ids s=some sf ∧
      runPrograms M D r ids s=some tf ∧ sf.mem=tf.mem := by
  obtain ⟨final,he⟩:=original_grid_executes M D r ids s
  exact ⟨final,final,he,(programs_execution_equal M D r ids s).symm.trans he,rfl⟩

theorem host_refinement {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (initial reference : State α)
    (he : Whole.host M D r initial=.ok (r.C,reference))
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧ final.mem=reference.mem := by
  obtain ⟨sf,tf,hs,ht,hm⟩ := programs_refinement M D r (List.range (programs r)) initial
  have href : sf=reference := by
    simpa [Whole.host,hdims,hcontiguous,hs] using he
  subst sf
  exact ⟨tf,by simp [host,hdims,hcontiguous,ht],hm.symm⟩

/-- Candidate grid execution is constructed without a numerical-law premise
or any input asserting a successful old/candidate execution. -/
theorem candidate_grid_executes {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat) (initial : State α) :
    ∃ final, runPrograms M D r ids initial=some final := by
  obtain ⟨_,final,_,he,_⟩ := programs_refinement M D r ids initial
  exact ⟨final,he⟩

/-- Whole candidate host success is a program theorem, independent of the six
numerical ports. Metadata assertions are the only host-success preconditions. -/
theorem host_executes {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (initial : State α)
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true) :
    ∃ final, host M D r initial=.ok (r.C,final) := by
  obtain ⟨final,he⟩ := candidate_grid_executes M D r (List.range (programs r)) initial
  exact ⟨final,by simp [host,hdims,hcontiguous,he]⟩

/-- The candidate also covers every output in any permutation of the complete
launch grid. The permutation is the explicit scheduling contract; output,
execution and memory-frame conclusions are not assumptions. -/
theorem candidate_grid_order_correct {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (r : Request) (initial : State α) (ids : List Nat)
    (order : ids.Perm (List.range (programs r))) (hCA : r.C≠r.A) (hCB : r.C≠r.B) :
    ∃ final, runPrograms M D r ids initial=some final ∧ Result Q r initial final := by
  obtain ⟨reference,hr,hresult⟩ := Whole.grid_order_correct M D Q law r initial ids order hCA hCB
  obtain ⟨sf,final,hs,he,hm⟩ := programs_refinement M D r ids initial
  have eqref : reference=sf := Option.some.inj (hr.symm.trans hs)
  subst sf
  refine ⟨final,he,?_⟩
  unfold Result at *
  simpa only [←hm] using hresult

theorem candidate_store_offset {α : Type} (r : Request) (s : State α)
    (i : FPValue.CIndex r.parameters) :
    Source.cSafeOffset r.parameters s i=
      safeCOffset r (FPValue.row r.parameters s i.1) (FPValue.col r.parameters s i.2.1) := by
  simp [Source.cSafeOffset,safeCOffset,safeCoordinate,Request.parameters]

/-- Unique ownership is proved for the actual clamped store address, not just
for the independent active-address formula. -/
theorem candidate_store_unique {α : Type} (r : Request) (s : State α)
    (p q : Nat) (hp : p<programs r) (hq : q<programs r)
    (i j : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters (instanceState r p s) i)
    (hj : FPValue.active r.parameters (instanceState r q s) j)
    (he : Source.cSafeOffset r.parameters (instanceState r p s) i=
      Source.cSafeOffset r.parameters (instanceState r q s) j) : p=q ∧ i=j := by
  have hiaddr : Source.cSafeOffset r.parameters (instanceState r p s) i=
      FPValue.cOffset r.parameters (instanceState r p s) i := by
    simp only [Source.cSafeOffset,FPValue.cOffset,hi.1,hi.2,if_true]
  have hjaddr : Source.cSafeOffset r.parameters (instanceState r q s) j=
      FPValue.cOffset r.parameters (instanceState r q s) j := by
    simp only [Source.cSafeOffset,FPValue.cOffset,hj.1,hj.2,if_true]
  rw [hiaddr,hjaddr] at he
  exact Whole.source_store_unique r s p q hp hq i j hi hj he

#axiomsClean candidate_grid_executes
#axiomsClean host_executes
#axiomsClean candidate_grid_order_correct
#axiomsClean candidate_store_unique
#axiomsClean programs_refinement
#axiomsClean host_refinement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
