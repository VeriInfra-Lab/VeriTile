import bench.repairs.tritonbench_g.streamk_matmul.Proofs.Envelope

namespace VeriTile.Bench.Repairs.StreamK.Checked.Host
open Contract TensorBinding Whole001 VeriTile.Torch
noncomputable section
set_option maxHeartbeats 3000000

structure Prepared (α : Type) where
  output : Tensor
  scratch : Option Tensor
  read : ValueRead α

abbrev Allocator (α : Type) := List Nat → Device → ValueRead α → Option (Tensor × ValueRead α)

/-- The actual allocation order: C, optional scratch, then the launch sequence.
Empty output returns immediately after allocating C. -/
def prepare (allocate : Allocator α) (a b : Tensor) (config : Config) (before : ValueRead α) : Option (Prepared α) := do
  let (out,read) ← allocate [rows a.metadata,columns b.metadata] a.metadata.device before
  if Empty a.metadata b.metadata then
    return ⟨out,none,read⟩
  else if 0<(Plan.select (plan a.metadata b.metadata config)).streamTiles then
    let (scratch,ready) ← allocate [config.programs.toNat,config.blockM,config.blockN] a.metadata.device read
    return ⟨out,some scratch,ready⟩
  else return ⟨out,none,read⟩

def base (t : Tensor) := t.metadata.dataPtr.toNat

def Outside (t : Tensor) (address : Nat) : Prop := address<base t ∨ base t+4*t.metadata.numel≤address
def Separate (t u : Tensor) : Prop :=
  base t+4*t.metadata.numel≤u.storageBase ∨ u.storageBase+4*u.storageElements≤base t

def OutsideScratch (p : Prepared α) (address : Nat) : Prop :=
  match p.scratch with | none => True | some s => Outside s address

/-- Ordinary allocator facts about the returned objects and newly allocated
byte windows. No source arithmetic bound or kernel result occurs here. -/
structure Allocation (a b : Tensor) (config : Config) (before : ValueRead α) (p : Prepared α) : Prop where
  outputLaws : TensorBinding.Laws p.output
  outputShape : p.output.metadata.shape=[rows a.metadata,columns b.metadata]
  outputDevice : p.output.metadata.device=a.metadata.device
  outputContiguous : p.output.metadata.isContiguous=Bool.true
  outputPacked : p.output.storageBase=base p.output ∧ p.output.storageElements=p.output.metadata.numel
  outputWindow : base p.output+4*p.output.metadata.numel≤2^64
  outputFreshA : Separate p.output a
  outputFreshB : Separate p.output b
  scratchPresent : p.scratch.isSome=decide (¬Empty a.metadata b.metadata ∧ 0<(Plan.select (plan a.metadata b.metadata config)).streamTiles)
  scratchProperties : ∀ s,p.scratch=some s →
    TensorBinding.Laws s ∧ s.metadata.shape=[config.programs.toNat,config.blockM,config.blockN] ∧
    s.metadata.device=a.metadata.device ∧ s.metadata.isContiguous=Bool.true ∧
    base s+4*s.metadata.numel≤2^64 ∧ Separate s a ∧ Separate s b ∧ Separate s p.output
  frame : ∀ address,Outside p.output address → OutsideScratch p address → p.read address=before address

def initial (zero : α) (a b : Tensor) (config : Config) (profile : Profile config)
    (prepared : Prepared α) : Source.State α (parameters a b config profile) 0 where
  inputA := fun i => prepared.read (base a+4*i.val)
  inputB := fun i => prepared.read (base b+4*i.val)
  output := fun i => prepared.read (base prepared.output+4*i.val)
  scratch := fun i => match prepared.scratch with
    | none => zero
    | some s => prepared.read (base s+4*i.val)
  exterior := Fin.elim0

def returned (ops : Source.Instructions α) (a b : Tensor) (config : Config) (profile : Profile config)
    (prepared : Prepared α)
    (first : Schedule.Order (Source.firstWave (parameters a b config profile) ops).length)
    (reduce : Schedule.CTAInterleaving (parameters a b config profile))
    (last : Schedule.Order (Source.fullWave (parameters a b config profile) ops).length) :=
  (prepared.output,Schedule.call (parameters a b config profile) ops first reduce last
    (initial ops.zero a b config profile prepared))

instance (a b : TensorMetadata) (c : Config) : Decidable (InputContract a b c) :=
  decidable_of_iff _ (contract_iff a b c).symm

/-- Ordinary compiler/device dispatch, given the actual A/B/C/scratch objects,
unchanged launch configuration, kernel identity and exact grid extent. -/
abbrev Dispatch (α : Type) := Tensor → Tensor → Prepared α → Config → Plan.Launch → Option Unit

def submit (dispatch : Dispatch α) (a b : Tensor) (config : Config) (p : Prepared α) : List Plan.Launch → Option Unit
  | [] => some ()
  | launch::tail => (dispatch a b p config launch).bind fun _ => submit dispatch a b config p tail

def Ready (dispatch : Dispatch α) (a b : Tensor) (config : Config) (p : Prepared α) : Prop :=
  ∀ launch∈Plan.launches (plan a.metadata b.metadata config),dispatch a b p config launch=some ()

theorem submit_ready (dispatch : Dispatch α) (a b : Tensor) (config : Config) (p : Prepared α)
    (launches : List Plan.Launch) (ready : ∀ launch∈launches,dispatch a b p config launch=some ()) :
    submit dispatch a b config p launches=some () := by
  induction launches with
  | nil => rfl
  | cons launch tail ih =>
    simp only [submit,ready launch (by simp),Option.bind_some]
    exact ih (fun l hl => ready l (by simp [hl]))

def call (ops : Source.Instructions α) (allocate : Allocator α) (dispatch : Dispatch α) (a b : Tensor) (config : Config)
    (before : ValueRead α)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (Source.firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (Source.fullWave (parameters a b config profile) ops).length) :
    Option (Tensor × PSigma (fun profile : Profile config => Source.State α (parameters a b config profile) 0)) :=
  if h : InputContract a.metadata b.metadata config then
    (prepare allocate a b config before).bind fun prepared =>
      (submit dispatch a b config prepared (Plan.launches (plan a.metadata b.metadata config))).map fun _ =>
        let schedule := schedules h.profile
        let result := returned ops a b config h.profile prepared schedule.1 schedule.2.1 schedule.2.2
        (result.1,⟨h.profile,result.2⟩)
  else none

theorem output_numel (a b : Tensor) (config : Config) (before : ValueRead α) (p : Prepared α)
    (law : Allocation a b config before p) : p.output.metadata.numel=rows a.metadata*columns b.metadata := by
  simp [TensorMetadata.numel,law.outputShape]

theorem input_outside (input out : Tensor) (laws : TensorBinding.Laws input)
    (separate : Separate out input) (i : Nat) (hi : i < input.metadata.numel) :
    Outside out (physicalAddress input i) := by
  have bound := laws.bounds i hi
  unfold Outside Separate physicalAddress at *
  omega

theorem prepared_inputs (zero : α) (read : ValueRead α) (a b : Tensor) (config : Config)
    (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b) (h : InputContract a.metadata b.metadata config)
    (prepared : Prepared α) (law : Allocation a b config read prepared) :
    (initial zero a b config h.profile prepared).inputA=left read a b config h.profile ∧
    (initial zero a b config h.profile prepared).inputB=right read a b config h.profile := by
  have bound : Inputs prepared.read a b config h.profile (initial zero a b config h.profile prepared) := ⟨fun _ => rfl,fun _ => rfl⟩
  obtain ⟨ha,hb⟩ := source_inputs prepared.read a b config la lb h _ bound
  refine ⟨ha.trans ?_,hb.trans ?_⟩
  · funext i
    have hi : i.val<a.metadata.numel := by rw [matrix_numel a.metadata h.inputs.2.2.1]; exact i.isLt
    apply law.frame
    · exact input_outside a prepared.output la law.outputFreshA i.val hi
    · cases hs : prepared.scratch with
      | none => simp [OutsideScratch,hs]
      | some s => simpa [OutsideScratch,hs] using input_outside a s la (law.scratchProperties s hs).2.2.2.2.2.1 i.val hi
  · funext i
    have hi : i.val<b.metadata.numel := by
      rw [matrix_numel b.metadata h.inputs.2.2.2.1,←h.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2]
      exact i.isLt
    apply law.frame
    · exact input_outside b prepared.output lb law.outputFreshB i.val hi
    · cases hs : prepared.scratch with
      | none => simp [OutsideScratch,hs]
      | some s => simpa [OutsideScratch,hs] using input_outside b s lb (law.scratchProperties s hs).2.2.2.2.2.2.1 i.val hi

theorem empty_stream (a b : Tensor) (config : Config) (profile : Profile config)
    (empty : Empty a.metadata b.metadata) : (Plan.select (plan a.metadata b.metadata config)).streamTiles=0 := by
  have bounded := Plan.selected_bounded (plan a.metadata b.metadata config)
  have mzero : VeriTile.Bench.TritonBenchG.StreamK.ceilDiv 0 config.blockM=0 := by
    unfold VeriTile.Bench.TritonBenchG.StreamK.ceilDiv
    exact Nat.div_eq_of_lt (by have hp := profile.1.1; omega)
  have nzero : VeriTile.Bench.TritonBenchG.StreamK.ceilDiv 0 config.blockN=0 := by
    unfold VeriTile.Bench.TritonBenchG.StreamK.ceilDiv
    exact Nat.div_eq_of_lt (by have hp := profile.2.1.1; omega)
  rcases empty with hm|hn
  · simp [Plan.tiles,plan,hm,mzero] at bounded
    simpa [plan,hm] using bounded
  · simp [Plan.tiles,plan,hn,nzero] at bounded
    simpa [plan,hn] using bounded

theorem scratch_present (a b : Tensor) (config : Config) (profile : Profile config)
    (before : ValueRead α) (p : Prepared α) (law : Allocation a b config before p) :
    p.scratch.isSome=decide (0<(Plan.select (plan a.metadata b.metadata config)).streamTiles) := by
  rw [law.scratchPresent]
  by_cases empty : Empty a.metadata b.metadata
  · simp [empty,empty_stream a b config profile empty]
  · simp [empty]

theorem call_prepared (ops : Source.Instructions α) (allocate : Allocator α) (dispatch : Dispatch α) (a b : Tensor) (config : Config)
    (before : ValueRead α) (h : InputContract a.metadata b.metadata config)
    (p : Prepared α) (hp : prepare allocate a b config before=some p)
    (ready : Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (Source.firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (Source.fullWave (parameters a b config profile) ops).length) :
    call ops allocate dispatch a b config before schedules=
      some (p.output,⟨h.profile,(returned ops a b config h.profile p
        (schedules h.profile).1 (schedules h.profile).2.1 (schedules h.profile).2.2).2⟩) := by
  simp only [call,dif_pos h,hp,Option.bind_some,submit_ready dispatch a b config p _ ready,Option.map_some]
  rfl

#axiomsClean prepared_inputs
#specNonCircular Allocation avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Reference.matrixProduct]
end
end VeriTile.Bench.Repairs.StreamK.Checked.Host
