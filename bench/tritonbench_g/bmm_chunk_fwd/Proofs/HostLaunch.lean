import bench.tritonbench_g.bmm_chunk_fwd.Proofs.HostGrid
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Mathematical_RealContract
import VeriTile.Triton.Memory.FlattenFrame

/-! Construct the whole ordinary launch from proved executions. All pins and
bounds concern initial inputs. The output frame is precisely the active tiles;
skipped causal tiles retain their original arbitrary torch.empty cells. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

def active (h : Call) (c : Config) (p : PID h c) (i : TileIndex [c.M,c.N]) : Prop :=
  (RealViews.io Bool.false c.M c.N c.K h.parameters).outputMask (p0 p) (p1 p) (p2 p) i
def offset (h : Call) (c : Config) (p : PID h c) (i : TileIndex [c.M,c.N]) :=
  RealViews.outputOffset c.M c.N h.parameters (p0 p) (p1 p) (p2 p) i
def value (h : Call) (c : Config) (x : Inputs h) (p : PID h c)
    (i : TileIndex [c.M,c.N]) :=
  expected h x (p1 p) (p2 p / h.shape.groups) (p2 p % h.shape.groups)
    (RealViews.row c.M c.N h.parameters (p0 p) i.1)
    (RealViews.column c.N h.parameters (p0 p) i.2.1)
def writes (h : Call) (c : Config) (p : PID h c) : WriteFootprint :=
  WriteFootprint.activeTileImage h.Out (offset h c p) (active h c p)

def Pinned (h : Call) (c : Config) (x : Inputs h) (s : BlockState) : Prop :=
  ∀ p : PID h c, RealIO.Pins (RealViews.io Bool.false c.M c.N c.K h.parameters)
    (s.withGridIndex p) (tileInputs h c x (p0 p) (p1 p) (p2 p))

theorem program_executes (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s) (p : PID h c) :
    ∃ t, exec (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel
        (s.withGridIndex p) = some t ∧
      (∀ i, active h c p i → t.mem h.Out (offset h c p i) = MemCell.real (value h c x p i)) ∧
      (∀ r a, ¬ writes h c p (r,a) → t.mem r a = s.mem r a) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped (s.withGridIndex p)
    c.M c.N c.K h.parameters (configuration_positive c hc).2.2
    (output_distinct h c _ _ _)
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    have hv' := hv i hi
    rw [RealViews.formula_of_pins Bool.false (s.withGridIndex p) c.M c.N c.K
      h.parameters _ (hp p) i hi] at hv'
    exact hv'
  · intro r a hn
    apply hf r a
    intro i hi heq
    exact hn ⟨heq.1.symm, i, hi, heq.2⟩

theorem frames_exist (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s) :
    ∃ frames : Kernel.GridFrames
        (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel (h.grid c) s,
      (∀ p, (frames p).writes = writes h c p) ∧
      (∀ p i, active h c p i → (frames p).final.mem h.Out (offset h c p i) =
        MemCell.real (value h c x p i)) := by
  classical
  choose t he hv hf using program_executes h c hc swapped x s hp
  let frames : Kernel.GridFrames
      (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel (h.grid c) s :=
    fun p => ⟨t p, writes h c p, he p, fun r a hn => (hf p r a hn).symm⟩
  exact ⟨frames, fun _ => rfl, hv⟩

theorem writes_disjoint (h : Call) (c : Config) (hc : c ∈ configurations)
    (p q : PID h c) (hne : p ≠ q) :
    WriteFootprint.disjoint (writes h c p) (writes h c q) := by
  intro addr ⟨_, i, hi, he⟩ ⟨_, j, hj, he'⟩
  exact hne (output_owner h c (configuration_positive c hc).1
    (configuration_positive c hc).2.1 p q i j hi hj (he.trans he'.symm))

/-- Each real execution preserves all input storage, not just the inputs of its
own tile. Hence any other program's input pins remain stable. -/
theorem program_input_stability (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s)
    (p : PID h c) (r : RegionName) (hr : r ≠ h.Out) :
    ∃ t, exec (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel
      (s.withGridIndex p) = some t ∧ ∀ a, t.mem r a = s.mem r a := by
  obtain ⟨t, he, _, hf⟩ := program_executes h c hc swapped x s hp p
  exact ⟨t, he, fun a => hf r a (fun hw => hr hw.1)⟩

def flatWrites (h : Call) (c : Config) (A : FlatAlloc) (p : PID h c) : WriteFootprint :=
  WriteFootprint.activeTileImage A.flat (fun i => A.addr h.Out (offset h c p i)) (active h c p)

/-- Finite initial input windows and the actual fresh output allocation. The
zero undef condition is interpreter plumbing, not a zero-filled allocation. -/
structure Allocated (h : Call) (c : Config) (s : BlockState) (A : FlatAlloc) : Prop where
  disjoint : A.Disjoint
  regions : A.regions = [h.A, h.B, Region.cast h.Seq, h.Out]
  closed : ∀ r, r ∉ A.regions → A.extent r = 0
  outputExtent : A.extent h.Out = h.shape.outputSize
  undef : s.undef = fun _ _ => 0
  inputs : ∀ p : PID h c, RealIO.Bounds
    (RealViews.io Bool.false c.M c.N c.K h.parameters) (s.withGridIndex p) A.extent

theorem flat_frames_exist (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s)
    (A : FlatAlloc) (ha : Allocated h c s A) :
    ∃ frames : Kernel.GridFrames
        (A.flattenKernel (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel)
        (h.grid c) (A.flattenState s),
      (∀ p, (frames p).writes = flatWrites h c A p) ∧
      (∀ p i, active h c p i → (frames p).final.mem A.flat (A.addr h.Out (offset h c p i)) =
        MemCell.real (value h c x p i)) := by
  classical
  choose t he hv hf using program_executes h c hc swapped x s hp
  have heflat (p : PID h c) :
      exec (A.flattenKernel (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel)
        ((A.flattenState s).withGridIndex p) = some (A.flattenState (t p)) := by
    change exec _ (A.flattenState (s.withGridIndex p)) = _
    rw [A.exec_flatten ha.disjoint ha.closed _ _
      (RealSafety.safe swapped _ c.M c.N c.K h.parameters A.extent
        (RealContract.bounds_of_views Bool.false _ c.M c.N c.K h.parameters A.extent (ha.inputs p)))
      (RealSafety.flatten_ok swapped c.M c.N c.K h.parameters) ha.undef, he p]
    rfl
  have hfflat (p : PID h c) (r : RegionName) (a : Nat)
      (hn : ¬ flatWrites h c A p (r,a)) :
      (A.flattenState (t p)).mem r a = (A.flattenState s).mem r a := by
    apply A.flattenState_frame s (t p) (fun r a => writes h c p (r,a)) (hf p) r a
    by_cases hr : r = A.flat
    · right
      intro region off ⟨hreg, i, hi, hoff⟩ heq
      apply hn
      dsimp only at hreg hoff
      rw [hreg, ← hoff] at heq
      exact ⟨hr, i, hi, heq.symm⟩
    · exact Or.inl hr
  let frames : Kernel.GridFrames
      (A.flattenKernel (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel)
      (h.grid c) (A.flattenState s) := fun p =>
    ⟨A.flattenState (t p), flatWrites h c A p, heflat p, fun r a hn => (hfflat p r a hn).symm⟩
  refine ⟨frames, fun _ => rfl, ?_⟩
  intro p i hi
  change (A.flattenState (t p)).mem A.flat (A.addr h.Out (offset h c p i)) = _
  rw [A.flattenState_mem_addr ha.disjoint (t p)
    (by rw [ha.regions]; simp) (by rw [ha.outputExtent]; exact output_bound h c p i hi), hv p i hi]
  rfl

def Written (h : Call) (c : Config) (A : FlatAlloc) (r : RegionName) (a : Nat) : Prop :=
  ∃ p : PID h c, flatWrites h c A p (r,a)

/-- Full finite-memory launch result, successful ordinary launch and every-cell
frame. This constructs the launch witness; it is not a launch assumption. -/
theorem launch (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s)
    (A : FlatAlloc) (ha : Allocated h c s A) :
    ∃ final, ∃ _ : Kernel.GridLaunchedOrdinary
      (A.flattenKernel (h.kernel c swapped).eraseDType.toAlgKernel)
      (h.grid c) (A.flattenState s) final,
      (∀ p i, active h c p i → final.readMem A.flat (A.addr (returned h).region (offset h c p i)) =
        value h c x p i) ∧
      (∀ r a, ¬ Written h c A r a → final.mem r a = (A.flattenState s).mem r a) := by
  obtain ⟨frames, hw, hv⟩ := flat_frames_exist h c hc swapped x s hp A ha
  have hd : Kernel.GridWritesDisjoint frames := by
    intro p q hpq addr hp hq
    rw [hw p] at hp
    rw [hw q] at hq
    rcases hp with ⟨_, i, hi, hei⟩
    rcases hq with ⟨_, j, hj, hej⟩
    have he := Nat.add_left_cancel (hei.trans hej.symm)
    exact hpq (output_owner h c (configuration_positive c hc).1
      (configuration_positive c hc).2.1 p q i j hi hj he)
  have hk : (h.kernel c swapped).eraseDType =
      RealProgram.kernel swapped c.M c.N c.K h.parameters :=
    RealProgram.storage_erasure h.aStorage h.bStorage swapped
      (outputDType h.aStorage h.requestedOutput) c.M c.N c.K h.parameters
  rw [hk]
  refine ⟨Kernel.mergeFrames (h.grid c) (A.flattenState s) frames, ⟨frames, hd, rfl⟩, ?_, ?_⟩
  · intro p i hi
    have hm := Kernel.mergeFrames_mem_written hd p A.flat (A.addr h.Out (offset h c p i))
      (by rw [hw]; exact ⟨rfl, i, hi, rfl⟩)
    simp only [returned, BlockState.readMem, hm, hv p i hi]
    rfl
  · intro r a hn
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨p, hp⟩
    apply hn
    exact ⟨p, by rwa [hw p] at hp⟩

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
