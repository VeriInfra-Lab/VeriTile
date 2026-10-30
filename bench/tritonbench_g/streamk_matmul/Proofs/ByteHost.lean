import bench.tritonbench_g.streamk_matmul.Proofs.Host
import bench.tritonbench_g.streamk_matmul.Proofs.ByteExecution

namespace VeriTile.Bench.Repairs.StreamK.Checked.ByteHost
open Contract TensorBinding Whole001 Source ByteMemory
noncomputable section
set_option maxHeartbeats 4000000

def layout (a b : Tensor) (config : Config) (profile : Profile config) (p : Host.Prepared α) :
    ByteExecution.Layout (parameters a b config profile) where
  outputBase := Host.base p.output
  scratchBase := (p.scratch.map Host.base).getD 0
  scratchActive := decide (0<(Plan.select (plan a.metadata b.metadata config)).streamTiles)
  scratch_eq := rfl

structure Allocation (a b : Tensor) (config : Config) (before ready : Read) (p : Host.Prepared Word) : Prop where
  tensors : Host.Allocation a b config (fun address => readWord before address) p
  words : ∀ address,p.read address=readWord ready address
  frame : ∀ address,Host.Outside p.output address → Host.OutsideScratch p address → ready address=before address

theorem layout_laws_general (a b : Tensor) (config : Config) (profile : Profile config)
    (before : ValueRead α) (p : Host.Prepared α) (law : Host.Allocation a b config before p) :
    ByteExecution.Laws (layout a b config profile p) := by
  have size := Host.output_numel a b config _ p law
  have present := Host.scratch_present a b config profile _ p law
  let v := layout a b config profile p
  have hc : (ByteExecution.cView v).extent=p.output.metadata.numel := size.symm
  have bc : v.outputBase=Host.base p.output := rfl
  change ByteExecution.Laws v
  constructor
  · cases hs : p.scratch with
    | none =>
      have inactive : v.scratchActive=Bool.false := by
        change decide _=Bool.false
        rw [←present,hs]; rfl
      unfold Separate ByteExecution.cView ByteExecution.sView
      simp only [inactive,Bool.false_eq_true,ite_false,Nat.mul_zero,Nat.add_zero]
      right
      simp [v,layout,hs]
    | some s =>
      have properties := law.scratchProperties s hs
      have sn : s.metadata.numel=config.programs.toNat*config.blockM*config.blockN := by
        simp [VeriTile.Torch.TensorMetadata.numel,properties.2.1,Nat.mul_assoc]
      have separate := properties.2.2.2.2.2.2.2
      unfold Host.Separate at separate
      rw [law.outputPacked.1,law.outputPacked.2] at separate
      have active : v.scratchActive=Bool.true := by
        change decide _=Bool.true
        rw [←present,hs]; rfl
      change Separate (ByteExecution.cView v) (ByteExecution.sView v)
      unfold Separate ByteExecution.cView ByteExecution.sView
      simp only [active,ite_true]
      have sb : v.scratchBase=Host.base s := by simp [v,layout,hs]
      rw [sb]
      change Host.base p.output+4*(rows a.metadata*columns b.metadata)≤Host.base s ∨
        Host.base s+4*(config.programs.toNat*config.blockM*config.blockN)≤Host.base p.output
      rw [←size,←sn]
      exact separate.symm
  · change Host.base p.output+4*(rows a.metadata*columns b.metadata)≤2^64
    rw [←size]
    exact law.outputWindow
  · intro active
    change (layout a b config profile p).scratchActive=Bool.true at active
    have presentTrue : p.scratch.isSome=Bool.true := present.trans active
    cases hs : p.scratch with
    | none => simp [hs] at presentTrue
    | some s =>
      have properties := law.scratchProperties s hs
      have sn : s.metadata.numel=config.programs.toNat*config.blockM*config.blockN := by
        simp [VeriTile.Torch.TensorMetadata.numel,properties.2.1,Nat.mul_assoc]
      change (p.scratch.map Host.base).getD 0+4*(if (layout a b config profile p).scratchActive then _ else 0)≤2^64
      simp only [hs,Option.map_some,Option.getD_some,active,ite_true]
      change Host.base s+4*(config.programs.toNat*config.blockM*config.blockN)≤2^64
      rw [←sn]
      exact properties.2.2.2.2.1

theorem layout_laws (a b : Tensor) (config : Config) (profile : Profile config)
    (before ready : Read) (p : Host.Prepared Word) (law : Allocation a b config before ready p) :
    ByteExecution.Laws (layout a b config profile p) :=
  layout_laws_general a b config profile _ p law.tensors

theorem initial_render (a b : Tensor) (config : Config) (profile : Profile config) (zero : Word)
    (before ready : Read) (p : Host.Prepared Word) (law : Allocation a b config before ready p) :
    ByteExecution.rendered ready (layout a b config profile p) (Host.initial zero a b config profile p)=ready := by
  let v := layout a b config profile p
  let s := Host.initial zero a b config profile p
  have present := Host.scratch_present a b config profile _ p law.tensors
  have scratch : Bound ready (ByteExecution.sView v) (extend s.scratch) := by
    intro i hi
    cases hs : p.scratch with
    | none =>
      have inactive : v.scratchActive=Bool.false := by change decide _=Bool.false; rw [←present,hs]; rfl
      simp [ByteExecution.sView,inactive] at hi
    | some scratch =>
      have active : v.scratchActive=Bool.true := by change decide _=Bool.true; rw [←present,hs]; rfl
      have bound : i<config.programs.toNat*config.blockM*config.blockN := by simpa [ByteExecution.sView,active] using hi
      have hb : i<(parameters a b config profile).h.programs*(parameters a b config profile).h.bm*(parameters a b config profile).h.bn := bound
      rw [show extend s.scratch i=s.scratch ⟨i,hb⟩ from extend_at s.scratch ⟨i,hb⟩]
      simp only [s,Host.initial,hs]
      rw [law.words]
      simp [v,layout,ByteExecution.sView,hs,readWord,read32]
  have output : Bound ready (ByteExecution.cView v) (extend s.output) := by
    intro i hi
    have hb : i<(parameters a b config profile).h.m*(parameters a b config profile).h.n := hi
    rw [show extend s.output i=s.output ⟨i,hb⟩ from extend_at s.output ⟨i,hb⟩]
    simp only [s,Host.initial]
    rw [law.words]
    rfl
  change render (render ready (ByteExecution.sView v) (extend s.scratch)) (ByteExecution.cView v) (extend s.output)=ready
  rw [render_initial _ _ _ scratch,render_initial _ _ _ output]

def result (ops : Instructions Word) (a b : Tensor) (config : Config) (profile : Profile config)
    (p : Host.Prepared Word) (ready : Read)
    (first : Schedule.Order (firstWave (parameters a b config profile) ops).length)
    (reduce : Schedule.CTAInterleaving (parameters a b config profile))
    (last : Schedule.Order (fullWave (parameters a b config profile) ops).length) : Tensor × Read :=
  (p.output,(ByteExecution.call (layout a b config profile p) ops first reduce last
    (Host.initial ops.zero a b config profile p,ready)).2)

theorem result_render (ops : Instructions Word) (a b : Tensor) (config : Config) (profile : Profile config)
    (before ready : Read) (p : Host.Prepared Word) (law : Allocation a b config before ready p)
    (first : Schedule.Order (firstWave (parameters a b config profile) ops).length)
    (reduce : Schedule.CTAInterleaving (parameters a b config profile))
    (last : Schedule.Order (fullWave (parameters a b config profile) ops).length) :
    result ops a b config profile p ready first reduce last=
      (p.output,ByteExecution.rendered ready (layout a b config profile p)
        (Host.returned ops a b config profile p first reduce last).2) := by
  have exec := ByteExecution.call_render (layout a b config profile p)
    (layout_laws a b config profile before ready p law) ops first reduce last ready
    (Host.initial ops.zero a b config profile p)
  rw [initial_render a b config profile ops.zero before ready p law] at exec
  exact congrArg (fun t => (p.output,t.2)) exec

def call (ops : Instructions Word) (allocate : Host.Allocator Word) (dispatch : Host.Dispatch Word) (a b : Tensor) (config : Config)
    (before : Read) (bytesFor : Host.Prepared Word → Read)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (fullWave (parameters a b config profile) ops).length) : Option (Tensor × Read) :=
  if h : InputContract a.metadata b.metadata config then
    (Host.prepare allocate a b config (fun address => readWord before address)).bind fun p =>
      (Host.submit dispatch a b config p (Plan.launches (plan a.metadata b.metadata config))).map fun _ =>
        result ops a b config h.profile p (bytesFor p)
          (schedules h.profile).1 (schedules h.profile).2.1 (schedules h.profile).2.2
  else none

theorem call_prepared (ops : Instructions Word) (allocate : Host.Allocator Word) (dispatch : Host.Dispatch Word) (a b : Tensor) (config : Config)
    (before : Read) (bytesFor : Host.Prepared Word → Read) (h : InputContract a.metadata b.metadata config)
    (p : Host.Prepared Word) (prepared : Host.prepare allocate a b config (fun address => readWord before address)=some p)
    (ready : Host.Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (fullWave (parameters a b config profile) ops).length) :
    call ops allocate dispatch a b config before bytesFor schedules=some
      (result ops a b config h.profile p (bytesFor p)
        (schedules h.profile).1 (schedules h.profile).2.1 (schedules h.profile).2.2) := by
  simp only [call,dif_pos h,prepared,Option.bind_some,Host.submit_ready dispatch a b config p _ ready,Option.map_some]

/-- Every physical byte outside the two actually allocated writable regions
keeps its pre-call value. Inactive scratch has no footprint. -/
theorem rendered_frame (a b : Tensor) (config : Config) (profile : Profile config)
    (before ready : Read) (p : Host.Prepared Word) (law : Allocation a b config before ready p)
    (final : State Word (parameters a b config profile) 0) (address : Nat)
    (hc : Host.Outside p.output address) (hs : Host.OutsideScratch p address) :
    ByteExecution.rendered ready (layout a b config profile p) final address=before address := by
  let v := layout a b config profile p
  have outsideC : Outside (ByteExecution.cView v) address := by
    change address<Host.base p.output ∨ Host.base p.output+4*(rows a.metadata*columns b.metadata)≤address
    rw [←Host.output_numel a b config _ p law.tensors]
    exact hc
  have outsideS : Outside (ByteExecution.sView v) address := by
    cases hp : p.scratch with
    | none =>
      have inactive : v.scratchActive=Bool.false := by
        change decide _=Bool.false
        rw [←Host.scratch_present a b config profile _ p law.tensors,hp]
        rfl
      simp only [Outside,ByteExecution.sView,inactive,Bool.false_eq_true,ite_false,Nat.mul_zero,Nat.add_zero]
      right
      simp [v,layout,hp]
    | some scratch =>
      have properties := law.tensors.scratchProperties scratch hp
      have sn : scratch.metadata.numel=config.programs.toNat*config.blockM*config.blockN := by
        simp [VeriTile.Torch.TensorMetadata.numel,properties.2.1,Nat.mul_assoc]
      have active : v.scratchActive=Bool.true := by
        change decide _=Bool.true
        rw [←Host.scratch_present a b config profile _ p law.tensors,hp]
        rfl
      simp only [Outside,ByteExecution.sView,active,ite_true]
      have sb : v.scratchBase=Host.base scratch := by simp [v,layout,hp]
      rw [sb]
      change address<Host.base scratch ∨ Host.base scratch+4*(config.programs.toNat*config.blockM*config.blockN)≤address
      rw [←sn]
      simpa [Host.OutsideScratch,hp] using hs
  change render (render ready (ByteExecution.sView v) (extend final.scratch)) (ByteExecution.cView v) (extend final.output) address=before address
  rw [render_frame _ _ _ _ outsideC,render_frame _ _ _ _ outsideS]
  exact law.frame address hc hs

theorem input_frame (a b : Tensor) (config : Config) (profile : Profile config)
    (before ready : Read) (p : Host.Prepared Word) (law : Allocation a b config before ready p)
    (final : State Word (parameters a b config profile) 0) (address : Nat)
    (inside : (a.storageBase≤address ∧ address<a.storageBase+4*a.storageElements) ∨
      (b.storageBase≤address ∧ address<b.storageBase+4*b.storageElements)) :
    ByteExecution.rendered ready (layout a b config profile p) final address=before address := by
  apply rendered_frame a b config profile before ready p law final address
  · have ha := law.tensors.outputFreshA
    have hb := law.tensors.outputFreshB
    unfold Host.Separate Host.Outside at *
    omega
  · cases hs : p.scratch with
    | none => simp [Host.OutsideScratch,hs]
    | some scratch =>
      have ha := (law.tensors.scratchProperties scratch hs).2.2.2.2.2.1
      have hb := (law.tensors.scratchProperties scratch hs).2.2.2.2.2.2.1
      simp only [Host.OutsideScratch,hs]
      unfold Host.Separate Host.Outside at *
      omega

#axiomsClean call_prepared
#axiomsClean rendered_frame
#axiomsClean input_frame
#axiomsClean layout_laws
#axiomsClean initial_render
#axiomsClean result_render
end
end VeriTile.Bench.Repairs.StreamK.Checked.ByteHost
