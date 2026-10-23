import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Execution

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

/-- A simulation over arbitrary launch lists. Existing grouped-grid coverage,
unique ownership and full host results are reused after this memory refinement. -/
theorem programs_refinement {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat)
    (s t : State α) (hm : s.mem=t.mem) :
    ∃ sf tf, Whole.runPrograms M D r ids s=some sf ∧
      runPrograms M D r ids t=some tf ∧ sf.mem=tf.mem := by
  induction ids generalizing s t with
  | nil => exact ⟨s,t,rfl,rfl,hm⟩
  | cons pid rest ih =>
    obtain ⟨sm,hs,hsm,_⟩ := FPValue.correctness M D r.parameters .single
      (instanceState r pid s) (Whole.active_injective r (instanceState r pid s))
    obtain ⟨tm,ht,htm⟩ := Source.execution_memory M D r.parameters (instanceState r pid t)
    have hmid : sm.mem=tm.mem := by
      rw [hsm,htm]
      exact written_mem_congr M D r.parameters _ _ hm rfl
    obtain ⟨sf,tf,hsf,htf,hfinal⟩ := ih sm tm hmid
    refine ⟨sf,tf,?_,?_,hfinal⟩
    · change (FP.Structural.exec (backend M D) (FPValue.source r.parameters .single)
        (instanceState r pid s)).bind (Whole.runPrograms M D r rest)=some sf
      rw [hs]
      exact hsf
    · change (FP.Structural.exec (backend M D) (Source.kernel r.parameters)
        (instanceState r pid t)).bind (runPrograms M D r rest)=some tf
      rw [ht]
      exact htf

theorem host_refinement {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (initial reference : State α)
    (he : Whole.host M D r initial=.ok (r.C,reference))
    (hdims : r.inner=r.innerB) (hcontiguous : r.aContiguous=Bool.true) :
    ∃ final, host M D r initial=.ok (r.C,final) ∧ final.mem=reference.mem := by
  obtain ⟨sf,tf,hs,ht,hm⟩ := programs_refinement M D r (List.range (programs r)) initial initial rfl
  have href : sf=reference := by
    simpa [Whole.host,hdims,hcontiguous,hs] using he
  subst sf
  exact ⟨tf,by simp [host,hdims,hcontiguous,ht],hm.symm⟩

/-- Candidate grid execution is constructed without a numerical-law premise
or any input asserting a successful old/candidate execution. -/
theorem candidate_grid_executes {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (ids : List Nat) (initial : State α) :
    ∃ final, runPrograms M D r ids initial=some final := by
  obtain ⟨_,final,_,he,_⟩ := programs_refinement M D r ids initial initial rfl
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
  obtain ⟨sf,final,hs,he,hm⟩ := programs_refinement M D r ids initial initial rfl
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
for the old source's active-address normal form. -/
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
