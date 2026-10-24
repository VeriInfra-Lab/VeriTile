import bench.repairs.tritonbench_g.streamk_matmul.Proofs.Index
import bench.repairs.tritonbench_g.streamk_matmul.Proofs.ValueMemory

namespace VeriTile.Bench.Repairs.StreamK.Checked.ValueHost
open Contract TensorBinding Whole001 Source
noncomputable section
set_option maxHeartbeats 4000000

theorem initial_render (a b : Tensor) (config : Config) (profile : Profile config) (zero : α)
    (before : ValueRead α) (p : Host.Prepared α) (law : Host.Allocation a b config before p) :
    ValueMemory.rendered p.read (ByteHost.layout a b config profile p) (Host.initial zero a b config profile p)=p.read := by
  let v := ByteHost.layout a b config profile p
  let s := Host.initial zero a b config profile p
  have scratch : (if v.scratchActive then ValueMemory.render p.read v.scratchBase s.scratch else p.read)=p.read := by
    cases hs : p.scratch with
    | none =>
      have inactive : v.scratchActive=Bool.false := by
        change decide _=Bool.false
        rw [←Host.scratch_present a b config profile _ p law,hs]
        rfl
      simp only [inactive,Bool.false_eq_true,ite_false]
    | some scratch =>
      have active : v.scratchActive=Bool.true := by
        change decide _=Bool.true
        rw [←Host.scratch_present a b config profile _ p law,hs]
        rfl
      simp only [active,ite_true]
      apply ValueMemory.initial
      intro i
      simp [s,Host.initial,v,ByteHost.layout,hs]
  change ValueMemory.render _ v.outputBase s.output=p.read
  rw [scratch]
  exact ValueMemory.initial p.read v.outputBase s.output (fun _ => rfl)

def result (ops : Instructions α) (a b : Tensor) (config : Config) (profile : Profile config)
    (p : Host.Prepared α)
    (first : Schedule.Order (firstWave (parameters a b config profile) ops).length)
    (reduce : Schedule.CTAInterleaving (parameters a b config profile))
    (last : Schedule.Order (fullWave (parameters a b config profile) ops).length) : Tensor × ValueRead α :=
  (p.output,(ValueMemory.call (ByteHost.layout a b config profile p) ops first reduce last
    (Host.initial ops.zero a b config profile p,p.read)).2)

theorem result_render (ops : Instructions α) (a b : Tensor) (config : Config) (profile : Profile config)
    (before : ValueRead α) (p : Host.Prepared α) (law : Host.Allocation a b config before p)
    (first : Schedule.Order (firstWave (parameters a b config profile) ops).length)
    (reduce : Schedule.CTAInterleaving (parameters a b config profile))
    (last : Schedule.Order (fullWave (parameters a b config profile) ops).length) :
    result ops a b config profile p first reduce last=
      (p.output,ValueMemory.rendered p.read (ByteHost.layout a b config profile p)
        (Host.returned ops a b config profile p first reduce last).2) := by
  have exec := ValueMemory.call_render (ByteHost.layout a b config profile p)
    (ByteHost.layout_laws_general a b config profile before p law) ops first reduce last p.read
    (Host.initial ops.zero a b config profile p)
  rw [initial_render a b config profile ops.zero before p law] at exec
  exact congrArg (fun t => (p.output,t.2)) exec

def call (ops : Instructions α) (allocate : Host.Allocator α) (dispatch : Host.Dispatch α)
    (a b : Tensor) (config : Config) (before : ValueRead α)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (fullWave (parameters a b config profile) ops).length) : Option (Tensor × ValueRead α) :=
  if h : InputContract a.metadata b.metadata config then
    (Host.prepare allocate a b config before).bind fun p =>
      (Host.submit dispatch a b config p (Plan.launches (plan a.metadata b.metadata config))).map fun _ =>
        result ops a b config h.profile p
          (schedules h.profile).1 (schedules h.profile).2.1 (schedules h.profile).2.2
  else none

theorem call_prepared (ops : Instructions α) (allocate : Host.Allocator α) (dispatch : Host.Dispatch α)
    (a b : Tensor) (config : Config) (before : ValueRead α) (h : InputContract a.metadata b.metadata config)
    (p : Host.Prepared α) (prepared : Host.prepare allocate a b config before=some p)
    (ready : Host.Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (firstWave (parameters a b config profile) ops).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (fullWave (parameters a b config profile) ops).length) :
    call ops allocate dispatch a b config before schedules=some
      (result ops a b config h.profile p
        (schedules h.profile).1 (schedules h.profile).2.1 (schedules h.profile).2.2) := by
  simp only [call,dif_pos h,prepared,Option.bind_some,Host.submit_ready dispatch a b config p _ ready,Option.map_some]

theorem rendered_frame (a b : Tensor) (config : Config) (profile : Profile config)
    (before : ValueRead α) (p : Host.Prepared α) (law : Host.Allocation a b config before p)
    (final : State α (parameters a b config profile) 0) (address : Nat)
    (hc : Host.Outside p.output address) (hs : Host.OutsideScratch p address) :
    ValueMemory.rendered p.read (ByteHost.layout a b config profile p) final address=before address := by
  let v := ByteHost.layout a b config profile p
  have outsideC : address<v.outputBase ∨ v.outputBase+4*((parameters a b config profile).h.m*(parameters a b config profile).h.n)≤address := by
    change address<Host.base p.output ∨ Host.base p.output+4*(rows a.metadata*columns b.metadata)≤address
    rw [←Host.output_numel a b config _ p law]
    exact hc
  change ValueMemory.render _ v.outputBase final.output address=before address
  rw [ValueMemory.frame _ _ _ _ outsideC]
  have scratch : (if v.scratchActive then ValueMemory.render p.read v.scratchBase final.scratch else p.read) address=p.read address := by
    cases hp : p.scratch with
    | none =>
      have inactive : v.scratchActive=Bool.false := by
        change decide _=Bool.false
        rw [←Host.scratch_present a b config profile _ p law,hp]
        rfl
      simp only [inactive,Bool.false_eq_true,ite_false]
    | some scratch =>
      have properties := law.scratchProperties scratch hp
      have sn : scratch.metadata.numel=config.programs.toNat*config.blockM*config.blockN := by
        simp [VeriTile.Torch.TensorMetadata.numel,properties.2.1,Nat.mul_assoc]
      split
      · apply ValueMemory.frame
        have sb : v.scratchBase=Host.base scratch := by simp [v,ByteHost.layout,hp]
        rw [sb]
        change address<Host.base scratch ∨ Host.base scratch+4*(config.programs.toNat*config.blockM*config.blockN)≤address
        rw [←sn]
        simpa [Host.OutsideScratch,hp] using hs
      · rfl
  exact scratch.trans (law.frame address hc hs)

theorem input_frame (a b : Tensor) (config : Config) (profile : Profile config)
    (before : ValueRead α) (p : Host.Prepared α) (law : Host.Allocation a b config before p)
    (final : State α (parameters a b config profile) 0) (address : Nat)
    (inside : (a.storageBase≤address ∧ address<a.storageBase+4*a.storageElements) ∨
      (b.storageBase≤address ∧ address<b.storageBase+4*b.storageElements)) :
    ValueMemory.rendered p.read (ByteHost.layout a b config profile p) final address=before address := by
  apply rendered_frame a b config profile before p law final address
  · have ha := law.outputFreshA
    have hb := law.outputFreshB
    unfold Host.Separate Host.Outside at *
    omega
  · cases hs : p.scratch with
    | none => simp [Host.OutsideScratch,hs]
    | some scratch =>
      have ha := (law.scratchProperties scratch hs).2.2.2.2.2.1
      have hb := (law.scratchProperties scratch hs).2.2.2.2.2.2.1
      simp only [Host.OutsideScratch,hs]
      unfold Host.Separate Host.Outside at *
      omega

#axiomsClean result_render
#axiomsClean call_prepared
#axiomsClean rendered_frame
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.StreamK.Checked.ValueHost
