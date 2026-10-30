import bench.tritonbench_g.context_attn_bloom.Proofs.FPEpilogue
import bench.tritonbench_g.context_attn_bloom.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

structure Call where
  args : Host.Arguments
  regions : Regions
  input : Inputs

def atIndex {g : Grid} (s : State α) (idx : GridIndex g) : State α :=
  {s with pids := idx.toPids, numPids := g.toNumPids}

def runGrid (c : Call) (base : Algebra α) (D : Dot α) :
    List (GridIndex (Host.grid c.args)) → State α → Option (State α)
  | [], s => some s
  | idx::rest, s => do
    let t ← exec ((withDot base D).withDefaultPrecision .fp32)
      (kernel (Host.layout c.args) c.regions (Host.scale c.args)) (atIndex s idx)
    runGrid c base D rest t

def allPrograms (c : Call) : List (GridIndex (Host.grid c.args)) := Finset.univ.toList

theorem allPrograms_complete (c : Call) (idx : GridIndex (Host.grid c.args)) :
    idx ∈ allPrograms c := by simp [allPrograms]

def Metadata (c : Call) (s : State α) : Prop :=
  (∀ b, (s.mem c.regions.start b).read .int = c.input.start b) ∧
  (∀ b, (s.mem c.regions.sequence b).read .int = c.input.sequence b) ∧
  (∀ b, (s.mem c.regions.prompt b).read .int = c.input.prompt b) ∧
  (∀ b, (s.mem c.regions.request b).read .int = c.input.request b) ∧
  (∀ j, (s.mem c.regions.tokens j).read .int = c.input.tokens j)

def laneActive (c : Call) (idx : GridIndex (Host.grid c.args))
    (i : RealIO.Lane (Host.layout c.args)) : Prop :=
  RealIO.active (Host.layout c.args) (Host.launchOf c.args idx) c.input i

def laneOffset (c : Call) (idx : GridIndex (Host.grid c.args))
    (i : RealIO.Lane (Host.layout c.args)) : Nat :=
  RealIO.write (Host.layout c.args) (Host.launchOf c.args idx) c.input i

def writes (c : Call) (order : List (GridIndex (Host.grid c.args))) (rg : RegionName) (o : Nat) : Prop :=
  ∃ idx ∈ order, ∃ i, laneActive c idx i ∧ rg=c.regions.out ∧ o=laneOffset c idx i

def Separate (c : Call) : Prop :=
  ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,
    c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.tokens] → c.regions.out≠rg

theorem active_matches (c : Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) :
    active (Host.layout c.args) c.regions (atIndex s idx) i ↔ laneActive c idx i := by
  unfold active length row atIndex laneActive RealIO.active Math.sequence Math.queryRow Host.launchOf
  simp only
  rw [hm.2.1,hm.2.2.1]
  simp [GridIndex.toPids,Host.grid,Grid.rank,Nat.mul_comm]

theorem offset_matches (c : Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) :
    offset (Host.layout c.args) c.regions (atIndex s idx) i = laneOffset c idx i := by
  unfold offset start row atIndex laneOffset RealIO.write AddressBounds.outputAddress Math.queryRow Host.launchOf
  simp only
  rw [hm.1]
  simp [GridIndex.toPids,Host.grid,Grid.rank,Nat.mul_comm]

theorem program_call (c : Call) (base : Algebra α) (D : Dot α)
    (s : State α) (hm : Metadata c s) (idx : GridIndex (Host.grid c.args)) :
    ∃ t, exec ((withDot base D).withDefaultPrecision .fp32)
      (kernel (Host.layout c.args) c.regions (Host.scale c.args)) (atIndex s idx)=some t ∧
      ∀ rg off, (¬∃ i, laneActive c idx i ∧ rg=c.regions.out ∧ off=laneOffset c idx i) →
        t.mem rg off=s.mem rg off := by
  obtain ⟨v,t,ht,hmt⟩ := program_success (Host.layout c.args) c.regions (Host.scale c.args)
    base D (Host.block_positive c.args) (atIndex s idx)
  refine ⟨t,ht,?_⟩
  intro rg off hmiss
  rw [hmt]
  apply scatter_frame
  intro i hi hr ho
  exact hmiss ⟨i,(active_matches c s hm idx i).mp hi,hr,ho.trans (offset_matches c s hm idx i)⟩

/-- Every CTA consumes the preceding CTA's memory. Kernel success and the
complete frame are conclusions for arbitrary scalar values, including NaN
and infinities. Only initial integer metadata and scratch ownership are used. -/
theorem grid_success_frame (c : Call) (base : Algebra α) (D : Dot α)
    (sep : Separate c) (order : List (GridIndex (Host.grid c.args)))
    (s : State α) (hm : Metadata c s) :
    ∃ t, runGrid c base D order s=some t ∧ Metadata c t ∧
      ∀ rg off, ¬writes c order rg off → t.mem rg off=s.mem rg off := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,hm,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨t,ht,hf⟩ := program_call c base D s hm idx
    have hr : ∀ rg, c.regions.out≠rg → ∀ off, t.mem rg off=s.mem rg off := by
      intro rg different off
      apply hf
      rintro ⟨i,hi,same,_⟩
      exact different same.symm
    have mt : Metadata c t := by
      rcases hm with ⟨a,b,d,e,f⟩
      refine ⟨?_,?_,?_,?_,?_⟩
      · intro i;rw [hr _ (sep _ (by simp))];exact a i
      · intro i;rw [hr _ (sep _ (by simp))];exact b i
      · intro i;rw [hr _ (sep _ (by simp))];exact d i
      · intro i;rw [hr _ (sep _ (by simp))];exact e i
      · intro i;rw [hr _ (sep _ (by simp))];exact f i
    obtain ⟨u,hu,mu,hfu⟩ := ih t mt
    refine ⟨u,by simp only [runGrid,ht,Option.bind_some];exact hu,mu,?_⟩
    intro rg off hmiss
    rw [hfu rg off (by rintro ⟨j,hj,i,hi,hr,ho⟩;exact hmiss ⟨j,by simp [hj],i,hi,hr,ho⟩)]
    apply hf
    rintro ⟨i,hi,hr,ho⟩
    exact hmiss ⟨idx,by simp,i,hi,hr,ho⟩

#axiomsClean grid_success_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
