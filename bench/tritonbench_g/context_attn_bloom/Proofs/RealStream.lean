import bench.tritonbench_g.context_attn_bloom.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealStream
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def raw (a : Host.Arguments) (r : Regions) : Kernel :=
  (Host.source a r Bool.false).eraseDType.toAlgKernel

def run (a : Host.Arguments) (r : Regions) :
    List (GridIndex (Host.grid a)) → BlockState → Option BlockState
  | [], s => some s
  | idx::rest,s => do
    let t ← exec (raw a r) (s.withGridIndex idx)
    run a r rest t

def flatRun (a : Host.Arguments) (r : Regions) (A : FlatAlloc) :
    List (GridIndex (Host.grid a)) → BlockState → Option BlockState
  | [],s => some s
  | idx::rest,s => do
    let t ← exec (A.flattenKernel (raw a r)) (s.withGridIndex idx)
    flatRun a r A rest t

def offset (a : Host.Arguments) (r : Regions) (s : BlockState)
    (idx : GridIndex (Host.grid a)) (i : RealIO.Lane (Host.layout a)) : Nat :=
  RealIO.write (Host.layout a) (Host.launchOf a idx) (inputs s r) i

def writes (a : Host.Arguments) (r : Regions) (s : BlockState)
    (jobs : List (GridIndex (Host.grid a))) (rg : RegionName) (off : Nat) : Prop :=
  ∃ idx ∈ jobs, ∃ i, Host.active a r s idx i ∧ rg=r.out ∧ off=offset a r s idx i

theorem admissible_transfer (a : Host.Arguments) (r : Regions) (A : FlatAlloc) (s t : BlockState)
    (valid : Host.Admissible a r A s) (same : inputs t r=inputs s r) : Host.Admissible a r A t := by
  refine { valid with metadata := ?_, packed := ?_, bounds := ?_ }
  · simpa only [same] using valid.metadata
  · simpa only [same] using valid.packed
  · simpa only [same] using valid.bounds

theorem inputs_frame (a : Host.Arguments) (r : Regions) (A : FlatAlloc) (s t : BlockState)
    (valid : Host.Admissible a r A s)
    (frame : ∀ rg, r.out≠rg → ∀ off, t.mem rg off=s.mem rg off) : inputs t r=inputs s r := by
  have sep := valid.separate
  unfold inputs
  congr 1
  all_goals
    funext off
    simp only [BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped]
    rw [frame]
  all_goals tauto

/-- One program is constructed from the current state. Its source execution,
independent formula, input stability, finite flat execution and frame are all
proved, rather than supplied as whole-program hypotheses. -/
theorem program (a : Host.Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (valid : Host.Admissible a r A s) (disjoint : A.Disjoint)
    (covered : ∀ rg, rg ∉ A.regions → A.extent rg=0)
    (zero : s.undef=(fun _ _ => 0)) (idx : GridIndex (Host.grid a)) :
    ∃ t, exec (raw a r) (s.withGridIndex idx)=some t ∧
      exec (A.flattenKernel (raw a r)) ((A.flattenState s).withGridIndex idx)=some (A.flattenState t) ∧
      inputs t r=inputs s r ∧ t.undef=s.undef ∧
      (∀ i, Host.active a r s idx i → t.readMem r.out (offset a r s idx i)=Host.result a r s idx i) ∧
      (∀ rg off, (¬∃ i, Host.active a r s idx i ∧ rg=r.out ∧ off=offset a r s idx i) →
        t.mem rg off=s.mem rg off) := by
  have prompt : 0≤(inputs (s.withGridIndex idx) r).prompt ((s.withGridIndex idx).pids 0) :=
    (valid.metadata _ (Host.batch_lt a idx)).2.1
  have unique : ∀ i j, EpilogueResult.activeLane (Host.layout a) r (s.withGridIndex idx) i →
      EpilogueResult.activeLane (Host.layout a) r (s.withGridIndex idx) j →
      EpilogueResult.offset (Host.layout a) r (s.withGridIndex idx) i=
        EpilogueResult.offset (Host.layout a) r (s.withGridIndex idx) j → i=j := by
    intro i j hi hj he
    exact (Host.unique_writer a r A s valid idx idx i j hi hj he).2
  obtain ⟨t,ht,hval,hframe⟩ := EpilogueResult.correct (Host.layout a) r .fp32 Bool.false
    (Host.scale a) (s.withGridIndex idx) (Host.block_positive a) prompt unique
  have ht' : exec (raw a r) (s.withGridIndex idx)=some t := ht
  have hf : ∀ rg off, (¬∃ i, Host.active a r s idx i ∧ rg=r.out ∧ off=offset a r s idx i) →
      t.mem rg off=s.mem rg off := by
    intro rg off hn
    apply hframe
    rintro ⟨i,hi,hr,ho⟩
    exact hn ⟨i,hi,hr.symm,ho.symm⟩
  have same : inputs t r=inputs s r := inputs_frame a r A s t valid (by
    intro rg different off
    apply hf
    rintro ⟨i,hi,eq,_⟩
    exact different eq.symm)
  have flatten := A.exec_flatten disjoint covered (raw a r) (s.withGridIndex idx)
    (WholeSafety.complete (Host.layout a) r .fp32 Bool.false (Host.scale a) (s.withGridIndex idx)
      (Host.block_positive a) prompt A.extent (by simpa only [Host.launch_grid,Host.inputs_grid] using valid.bounds idx))
    (FlattenShape.supported (Host.layout a) r .fp32 Bool.false (Host.scale a)) zero
  rw [ht',Option.map_some] at flatten
  refine ⟨t,ht',flatten,same,?_,hval,hf⟩
  exact stepStmts_undef _ (s.withGridIndex idx) t (FlattenShape.supported (Host.layout a) r .fp32 Bool.false (Host.scale a)) ht

/-- Complete sequential execution in one finite flat allocation. Every CTA
reads current memory, and immutable prepared inputs keep the independent
formula fixed. Earlier output writers are protected by global uniqueness. -/
theorem complete (a : Host.Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (valid : Host.Admissible a r A s) (disjoint : A.Disjoint)
    (covered : ∀ rg, rg ∉ A.regions → A.extent rg=0)
    (zero : s.undef=(fun _ _ => 0)) (jobs : List (GridIndex (Host.grid a))) :
    ∃ t, run a r jobs s=some t ∧ flatRun a r A jobs (A.flattenState s)=some (A.flattenState t) ∧
      inputs t r=inputs s r ∧ t.undef=s.undef ∧
      (∀ idx∈jobs, ∀ i, Host.active a r s idx i →
        t.readMem r.out (offset a r s idx i)=Host.result a r s idx i) ∧
      (∀ rg off, ¬writes a r s jobs rg off → t.mem rg off=s.mem rg off) := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl,rfl,rfl,by simp,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨t,ht,hflat,hinput,hu,hval,hframe⟩ := program a r A s valid disjoint covered zero idx
    obtain ⟨u,he,hef,hui,huu,hvals,hframes⟩ := ih t
      (admissible_transfer a r A s t valid hinput) (hu.trans zero)
    have activeEq : ∀ j i, Host.active a r t j i ↔ Host.active a r s j i := by
      intros; unfold Host.active;rw [hinput]
    have offEq : ∀ j i, offset a r t j i=offset a r s j i := by
      intros; unfold offset;rw [hinput]
    have resultEq : ∀ j i, Host.result a r t j i=Host.result a r s j i := by
      intros; unfold Host.result;rw [hinput]
    refine ⟨u,by simp only [run,ht,Option.bind_some];exact he,
      by simp only [flatRun,hflat,Option.bind_some];exact hef,hui.trans hinput,huu.trans hu,?_,?_⟩
    · intro j hj i hi
      by_cases later : j∈rest
      · simpa only [offEq,resultEq] using hvals j later i ((activeEq j i).mpr hi)
      · have eq : j=idx := by simpa [later] using hj
        subst j
        have stable := hframes r.out (offset a r t idx i) (by
          rintro ⟨k,hk,l,hl,hr,ho⟩
          have unique := (Host.unique_writer a r A s valid idx k i l hi
            ((activeEq k l).mp hl) (by simpa only [offEq] using ho)).1
          exact later (unique ▸ hk))
        simp only [offEq] at stable
        simpa only [BlockState.readMem,stable] using hval i hi
    · intro rg off miss
      rw [hframes rg off (by
        rintro ⟨j,hj,i,hi,hr,ho⟩
        exact miss ⟨j,by simp [hj],i,(activeEq j i).mp hi,hr,ho.trans (offEq j i)⟩)]
      apply hframe
      rintro ⟨i,hi,hr,ho⟩
      exact miss ⟨idx,by simp,i,hi,hr,ho⟩

#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealStream
