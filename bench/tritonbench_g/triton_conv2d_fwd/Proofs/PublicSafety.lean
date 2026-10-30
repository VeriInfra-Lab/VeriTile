import bench.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalInputs
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSafety
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultSafety

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural KernelNumeric KernelBinding
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem occurs_in_bounds (o : Operations A) (s : Shape) (f : Flags) (vals : Inputs A s)
    (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (r : RegionName) (off : Nat) (occurs : Occurs s r off) : off < bounds r := by
  obtain ⟨pos,hr,ha⟩ := occurs
  have h := hb.2.1 pos.1 pos.2 trivial
  change readAddress s pos.1 pos.2 < bounds (inputChannels s)[pos.1].buf at h
  simpa only [hr,ha] using h

theorem output_in_bounds (o : Operations A) (s : Shape) (f : Flags) (vals : Inputs A s)
    (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (x : Conv2d.Cell) (hx : s.Active x) : outputPointer s x < bounds "output" := by
  exact hb.2.2.1 ⟨0,by simp [program,memory,outputChannels]⟩
    (⟨x.group,hx.1⟩,⟨x.spatial,hx.2.1⟩,⟨x.feature,hx.2.2⟩,()) trivial

theorem source_bounds (o : Operations A) (s : Shape) (v : Valid s) (f : Flags)
    (vals : Inputs A s) (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (pid : Program) (hg : pid.2.2 < s.groups) (t : State (Payload A)) :
    KernelHalf.Access.Bounds (parameters s f) (atProgram pid t) bounds := by
  constructor
  · intro h hh w hw c hc i active
    exact occurs_in_bounds o s f vals bounds hb _ _ (source_input_occurs s v f pid hg t h w c i active)
  · intro h hh w hw c hc i active
    exact occurs_in_bounds o s f vals bounds hb _ _ (source_weight_occurs s v f pid hg t h w c hh hw i active)
  · intro i active
    have h := output_in_bounds o s f vals bounds hb (cell pid i) ((active_iff s v f pid hg t i).mp active)
    simpa only [output_address s v f] using h

@[simp] theorem invocation_initial (o : Operations A) (s : Shape) (f : Flags)
    (pid : Program) (t : State (Payload A)) :
    (invocation o s f pid).initial t=atProgram pid t := by rfl

@[simp] theorem invocation_run (o : Operations A) (s : Shape) (f : Flags)
    (pid : Program) (t : State (Payload A)) :
    (invocation o s f pid).run t=
      KernelGrid.program (algebra o) (halfDot o) (defaultDot o) s f pid t := by rfl

theorem schedule_run (o : Operations A) (s : Shape) (f : Flags)
    (work : List Program) (t : State (Payload A)) :
    runSchedule (work.map (invocation o s f)) t=
      KernelGrid.run (algebra o) (halfDot o) (defaultDot o) s f work t := by
  induction work generalizing t with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,runSchedule,KernelGrid.run,invocation_run,ih]

theorem invocation_safe (o : Operations A) (s : Shape) (v : Valid s) (f : Flags)
    (vals : Inputs A s) (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (pid : Program) (hg : pid.2.2 < s.groups) (t : State (Payload A)) :
    (invocation o s f pid).Safe bounds t := by
  have h := source_bounds o s v f vals bounds hb pid hg t
  change TraceSafe (KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f)
    bounds (KernelSource.kernel f.fp16 (parameters s f)) (atProgram pid t)
  by_cases hf : f.fp16=Bool.true
  · simpa [KernelGrid.backend,KernelSource.kernel,hf] using
      KernelHalf.Access.source_safe (algebra o) (halfDot o) (parameters s f) (atProgram pid t) bounds h
  · have hf' : f.fp16=Bool.false := Bool.eq_false_iff.mpr hf
    have h' : KernelDefault.Access.Bounds (parameters s f) (atProgram pid t) bounds :=
      ⟨h.input,h.weight,h.output⟩
    simpa [KernelGrid.backend,KernelSource.kernel,hf'] using
      KernelDefault.Access.source_safe (algebra o) (defaultDot o) (parameters s f) (atProgram pid t) bounds h'

theorem schedule_safe (o : Operations A) (s : Shape) (v : Valid s) (f : Flags)
    (vals : Inputs A s) (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (work : List Program) (legal : ∀ pid ∈ work, pid.2.2 < s.groups) (t : State (Payload A)) :
    ScheduleSafe bounds (work.map (invocation o s f)) t := by
  induction work generalizing t with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨invocation_safe o s v f vals bounds hb pid (legal pid (by simp)) t,
      fun u _ => ih (fun p hp => legal p (by simp [hp])) u⟩

theorem whole_safe (o : Operations A) (s : Shape) (v : Valid s) (f : Flags)
    (vals : Inputs A s) (bounds : RegionName → Nat) (hb : (program o s f).InBounds bounds vals)
    (t : State (Payload A)) : ScheduleSafe bounds ((program o s f).schedule vals) t := by
  apply schedule_safe o s v f vals bounds hb
  intro pid hp
  rcases pid with ⟨x,y,g⟩
  simp only [programs,List.mem_product,List.mem_range] at hp
  exact hp.2.2

theorem frame_of_exterior (o : Operations A) (s : Shape) (v : Valid s) (f : Flags)
    (vals : Inputs A s) (t u : State (Payload A))
    (frame : ∀ (r : RegionName) a, r≠"output" ∨ s.size≤a → u.mem r a=t.mem r a) :
    (program o s f).Frame vals t u := by
  intro r a outside
  apply frame r a
  by_cases hr : r="output"
  · subst r
    right
    by_contra h
    obtain ⟨j,hj⟩ := output_covers s v a (by omega)
    have bad := outside.1 ⟨0,by simp [program,memory,outputChannels]⟩ j trivial
    change ("output":RegionName)≠"output" ∨ a≠outputAddress s j at bad
    rcases bad with bad|bad
    · exact bad rfl
    · exact bad hj.symm
  · exact Or.inl hr

#axiomsClean whole_safe
end
end VeriTile.Bench.Repairs.Conv2d.ConvIO
