import bench.repairs.tritonbench_g.bmm_chunk_fwd.Kernels
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.HostReturned

/-! Ordered ordinary Real launches for the actual finite tuner trace. The
region-state witness is retained between launches, and its flat image is the
actual final state of the preceding launch. In particular, no flat state is
flattened for a second time. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.RealTrials
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 2000000

/-- Every entry executes an ordinary complete grid against the preceding
entry's actual flat final state. The list may be empty and may repeat entries. -/
inductive Runs (h : Host.Call) (A : FlatAlloc) :
    List Host.Config → BlockState → BlockState → Prop where
  | nil (s) : Runs h A [] s s
  | cons (c cs s middle final)
      (launch : Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (Kernels.source h c).eraseDType.toAlgKernel)
        (h.grid c) s middle))
      (rest : Runs h A cs middle final) : Runs h A (c::cs) s final

/-- Exact union of configuration-dependent active causal stores. -/
def Written (h : Host.Call) (cs : List Host.Config) (A : FlatAlloc)
    (r : RegionName) (off : Nat) : Prop :=
  ∃ c∈cs, Host.Written h c A r off

theorem runs_append (h : Host.Call) (A : FlatAlloc) (xs ys : List Host.Config)
    (s middle final : BlockState) (hx : Runs h A xs s middle)
    (hy : Runs h A ys middle final) : Runs h A (xs++ys) s final := by
  induction hx with
  | nil => exact hy
  | cons c cs s mid last launch rest ih =>
    exact .cons c (cs++ys) s mid final launch (ih hy)

/-- One launch constructs a named-region successor whose flat image equals
the complete ordinary flat-grid result. Input pins and undef survive the
merge, while the frame contains exactly the current configuration's stores. -/
theorem step (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (x : Host.Inputs h) (s : BlockState) (A : FlatAlloc)
    (hd : A.Disjoint) (regions : A.regions=[h.A,h.B,Region.cast h.Seq,h.Out])
    (closed : ∀ r, r∉A.regions → A.extent r=0)
    (out : A.extent h.Out=h.shape.outputSize) (undef : s.undef=fun _ _ => 0)
    (ht : Host.TensorInputs h x s A.extent) :
    ∃ t, Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (Kernels.source h c).eraseDType.toAlgKernel)
        (h.grid c) (A.flattenState s) (A.flattenState t)) ∧
      Host.TensorInputs h x t A.extent ∧ (t.undef=fun _ _ => 0) ∧
      (∀ r off, ¬Host.Written h c A r off →
        (A.flattenState t).mem r off=(A.flattenState s).mem r off) := by
  classical
  have hp := Host.tensor_pins h c x s A.extent ht
  obtain ⟨rf,rw,rv⟩ := Host.frames_exist h c hc Bool.false x s hp
  obtain ⟨ff,fw,fv⟩ := Host.flat_frames_exist h c hc Bool.false x s hp A
    ⟨hd,regions,closed,out,undef,Host.tensor_bounds h c x s A.extent ht out⟩
  have rd : Kernel.GridWritesDisjoint rf := by
    intro p q hpq
    rw [rw p,rw q]
    exact Host.writes_disjoint h c hc p q hpq
  have fd : Kernel.GridWritesDisjoint ff := by
    intro p q hpq addr hwp hwq
    rw [fw p] at hwp
    rw [fw q] at hwq
    rcases hwp with ⟨_,i,hi,hei⟩
    rcases hwq with ⟨_,j,hj,hej⟩
    have he := Nat.add_left_cancel (hei.trans hej.symm)
    exact hpq (Host.output_owner h c (Host.configuration_positive c hc).1
      (Host.configuration_positive c hc).2.1 p q i j hi hj he)
  let t := Kernel.mergeFrames (h.grid c) s rf
  have regionFrame (r : RegionName) (off : Nat)
      (hn : ¬∃ p, Host.writes h c p (r,off)) : t.mem r off=s.mem r off := by
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨p,hw⟩
    exact hn ⟨p,by rwa [rw p] at hw⟩
  have flatFrame (r : RegionName) (off : Nat) (hn : ¬Host.Written h c A r off) :
      (A.flattenState t).mem r off=(A.flattenState s).mem r off := by
    apply A.flattenState_frame s t (fun r off => ∃ p, Host.writes h c p (r,off))
      regionFrame r off
    by_cases hr : r=A.flat
    · right
      rintro region addr ⟨p,hreg,i,hi,hoff⟩ heq
      apply hn
      dsimp only at hreg hoff
      rw [hreg,←hoff] at heq
      exact ⟨p,hr,i,hi,heq.symm⟩
    · exact Or.inl hr
  have joined : A.flattenState t=Kernel.mergeFrames (h.grid c) (A.flattenState s) ff := by
    have mem : (A.flattenState t).mem=
        (Kernel.mergeFrames (h.grid c) (A.flattenState s) ff).mem := by
      funext r off
      by_cases hw : Host.Written h c A r off
      · rcases hw with ⟨p,hr,i,hi,hoff⟩
        dsimp only at hr hoff
        subst r
        rw [←hoff]
        rw [A.flattenState_mem_addr hd t (by rw [regions]; simp)
          (by rw [out]; exact Host.output_bound h c p i hi)]
        have rm := Kernel.mergeFrames_mem_written rd p h.Out (Host.offset h c p i)
          (by rw [rw p]; exact ⟨rfl,i,hi,rfl⟩)
        have fm := Kernel.mergeFrames_mem_written fd p A.flat (A.addr h.Out (Host.offset h c p i))
          (by rw [fw p]; exact ⟨rfl,i,hi,rfl⟩)
        rw [show t.mem h.Out (Host.offset h c p i)=_ from rm,rv p i hi,fm,fv p i hi]
        rfl
      · rw [flatFrame r off hw]
        symm
        apply Kernel.mergeFrames_mem_unwritten
        rintro ⟨p,hp⟩
        exact hw ⟨p,by rwa [fw p] at hp⟩
    exact congrArg (fun memory => { A.flattenState s with mem := memory }) mem
  have inputFrame (r : RegionName) (hr : r≠h.Out) (off : Nat) :
      t.mem r off=s.mem r off := regionFrame r off (by rintro ⟨p,hw⟩; exact hr hw.1)
  refine ⟨t,?_,{ ht with aPins := ?_, bPins := ?_, seqPins := ?_ },undef,flatFrame⟩
  · have hk : (Kernels.source h c).eraseDType=
        RealProgram.kernel Bool.false c.M c.N c.K h.parameters :=
      RealProgram.storage_erasure h.aStorage h.bStorage Bool.false
        (Host.outputDType h.aStorage h.requestedOutput) c.M c.N c.K h.parameters
    rw [hk,joined]
    exact ⟨⟨ff,fd,rfl⟩⟩
  · intro b r g k hb hr hg
    simpa only [BlockState.readMem,inputFrame _ ht.freshA] using ht.aPins b r g k hb hr hg
  · intro b r g k hb hr hg
    simpa only [BlockState.readMem,inputFrame _ ht.freshB] using ht.bPins b r g k hb hr hg
  · intro hs b r hb hr
    simpa only [BlockState.readMemValue,BlockState.readMemTyped,inputFrame _ ht.freshSeq]
      using ht.seqPins hs b r hb hr

/-- Arbitrarily many legal benchmark launches preserve the original input
bindings. Every cell outside their union footprint keeps its allocation value. -/
theorem execute (h : Host.Call) (cs : List Host.Config)
    (legal : ∀ c∈cs, c∈Host.configurations) (x : Host.Inputs h)
    (s : BlockState) (A : FlatAlloc)
    (hd : A.Disjoint) (regions : A.regions=[h.A,h.B,Region.cast h.Seq,h.Out])
    (closed : ∀ r, r∉A.regions → A.extent r=0)
    (out : A.extent h.Out=h.shape.outputSize) (undef : s.undef=fun _ _ => 0)
    (ht : Host.TensorInputs h x s A.extent) :
    ∃ t, Runs h A cs (A.flattenState s) (A.flattenState t) ∧
      Host.TensorInputs h x t A.extent ∧ (t.undef=fun _ _ => 0) ∧
      (∀ r off, ¬Written h cs A r off →
        (A.flattenState t).mem r off=(A.flattenState s).mem r off) := by
  induction cs generalizing s with
  | nil => exact ⟨s,.nil _,ht,undef,fun _ _ _ => rfl⟩
  | cons c cs ih =>
    obtain ⟨middle,launch,hm,hu,hf⟩ := step h c (legal c (by simp)) x s A
      hd regions closed out undef ht
    obtain ⟨t,rest,ht,tu,tf⟩ := ih (fun c hc => legal c (by simp [hc])) middle hu hm
    refine ⟨t,.cons c cs _ _ _ launch rest,ht,tu,?_⟩
    intro r off hn
    exact (tf r off (by rintro ⟨config,hc,hw⟩; exact hn ⟨config,by simp [hc],hw⟩)).trans
      (hf r off (by intro hw; exact hn ⟨c,by simp,hw⟩))

#axiomsClean step
#axiomsClean execute
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.RealTrials
