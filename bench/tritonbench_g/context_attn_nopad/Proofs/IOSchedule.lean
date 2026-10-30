import bench.tritonbench_g.context_attn_nopad.Proofs.IOBounds

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000
variable {α : Type} [Inhabited α]

theorem schedule_run (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (order : List (GridIndex c.grid)) (s : State α) :
    runSchedule (order.map (job c r o)) s=FPReference.runGrid c r o order s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,job_initial,FPReference.runGrid]
    congr 1
    funext t
    exact ih t

theorem successor_metadata (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (idx : GridIndex c.grid) (s t : State α) (metadata : FPHostSource.Metadata c r s)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence)
    (ran : (job c r o idx).run s=some t) : FPHostSource.Metadata c r t := by
  obtain ⟨u,he,hm⟩ := FPReference.program c.layout r c.scale o (by norm_num [Call.layout])
    (FPHost.atProgram c idx s)
  change exec (FP.ProfiledRewrite.engine o.algebra) (FPSymbolic.program c.layout r c.scale)
    (FPHost.atProgram c idx s)=some t at ran
  rw [he] at ran
  cases Option.some.inj ran
  have metadata' := (FPReference.grid_frame c r o [idx] s metadata startLabel lengthLabel).1
  change FPHostSource.Metadata c r (FPReference.tile c r o idx s) at metadata'
  constructor
  · intro b hb
    change (t.mem r.start b).read .int=_
    rw [hm]
    exact metadata'.start b hb
  · intro b hb
    change (t.mem r.sequence b).read .int=_
    rw [hm]
    exact metadata'.length b hb

theorem schedule_safe (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (xs : (io c r o).Inputs) (bounds : RegionName → Nat) (hb : (io c r o).InBounds bounds xs)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : FPHostSource.Metadata c r s) :
    ScheduleSafe bounds (order.map (job c r o)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons idx rest ih =>
    constructor
    · exact IOSourceSafety.program_safe c.layout r c.scale o (by norm_num [Call.layout])
        (FPHost.atProgram c idx s) bounds (source_bounds c r o xs bounds hb s metadata idx)
    · intro t ran
      exact ih t (successor_metadata c r o idx s t metadata startLabel lengthLabel ran)

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
