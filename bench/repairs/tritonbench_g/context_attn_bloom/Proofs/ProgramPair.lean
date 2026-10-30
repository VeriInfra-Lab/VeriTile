import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IndexCSE
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PublicNative
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

def candidateInvocation (g : Raw.Geometry) (r : Regions)
    (idx : GridIndex (Host.grid g.arguments)) : Invocation Native.Expr :=
  {KernelIO.invocation g r Native.scalar Native.dots idx with
    kernel:=IndexCSE.candidate (Host.layout g.arguments) r (Host.scale g.arguments)}

def candidateSchedule (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : (NativeIO.io g r extent).Inputs) : List (Invocation Native.Expr) :=
  (KernelIO.order g r extent xs).map (candidateInvocation g r)

/-- Only actual structural invocation schedules are implementation data. The
existing native IO, logical domain, WordISA and SM80 observation stay fixed. -/
structure Implementation (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) where
  schedule : (NativeIO.io g r extent).Inputs → List (Invocation Native.Expr)

def original (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    Implementation g r extent ops := ⟨(NativeIO.io g r extent).schedule⟩
def indexShared (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    Implementation g r extent ops := ⟨candidateSchedule g r extent⟩

/-- Both complete safe and successful metadata-selected source grids, every
native observed typed cell and both exact raw caller frames, on precisely the
unchanged PublicDomain.native (including its empty-grid branch). -/
def Equivalent {g : Raw.Geometry} {r : Regions} {extent : RegionBounds} {ops : Native.WordISA}
    (left right : Implementation g r extent ops) : Prop :=
  let io:=NativeIO.io g r extent
  let observe:=NativeIO.observe g r extent ops
  ∀ xs, PublicDomain.native g r extent xs → ∀ bounds, io.InBounds bounds xs → ∀ initial,
    io.InputBinding xs initial →
    ScheduleSafe bounds (left.schedule xs) initial ∧ ScheduleSafe bounds (right.schedule xs) initial ∧
    ∃ l rr, runSchedule (left.schedule xs) initial=some l ∧ runSchedule (right.schedule xs) initial=some rr ∧
      (∀ o j, io.outputMask o xs j →
        mapCell (observe o j) (mapCell (io.observe o j) (l.mem io.outputs[o].buf (io.write o xs j)))=
        mapCell (observe o j) (mapCell (io.observe o j) (rr.mem io.outputs[o].buf (io.write o xs j)))) ∧
      io.Frame xs initial l ∧ io.Frame xs initial rr

instance {g : Raw.Geometry} {r : Regions} {extent : RegionBounds} {ops : Native.WordISA} :
    Spec.ProgramSyntax (Implementation g r extent ops) where
  Statement := Implementation g r extent ops
  Signature := Unit
  signature := fun _=>()
  body := fun i=>[i]
  structural := some Equivalent
  sameContext := Eq

theorem invocation_run (g : Raw.Geometry) (r : Regions)
    (idx : GridIndex (Host.grid g.arguments)) (s : State Native.Expr) :
    (candidateInvocation g r idx).run s=(KernelIO.invocation g r Native.scalar Native.dots idx).run s := by
  change exec _ (IndexCSE.candidate _ _ _) ((KernelIO.invocation g r Native.scalar Native.dots idx).initial s)=
    exec _ (kernel _ _ _) ((KernelIO.invocation g r Native.scalar Native.dots idx).initial s)
  rw [ContextAttnBloomSource.selected_binding]
  exact (IndexCSE.execution _ _ _ _ _).symm

theorem invocation_safe (g : Raw.Geometry) (r : Regions)
    (idx : GridIndex (Host.grid g.arguments)) (bounds : RegionName → Nat) (s : State Native.Expr) :
    (KernelIO.invocation g r Native.scalar Native.dots idx).Safe bounds s ↔
      (candidateInvocation g r idx).Safe bounds s := by
  change TraceSafe _ bounds (kernel _ _ _) ((KernelIO.invocation g r Native.scalar Native.dots idx).initial s) ↔
    TraceSafe _ bounds (IndexCSE.candidate _ _ _) ((KernelIO.invocation g r Native.scalar Native.dots idx).initial s)
  rw [ContextAttnBloomSource.selected_binding]
  exact IndexCSE.trace_safe _ _ _ _ _ _

theorem schedule_run (g : Raw.Geometry) (r : Regions)
    (jobs : List (GridIndex (Host.grid g.arguments))) (s : State Native.Expr) :
    runSchedule (jobs.map (candidateInvocation g r)) s=
      runSchedule (jobs.map (KernelIO.invocation g r Native.scalar Native.dots)) s := by
  induction jobs generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    simp only [List.map_cons,runSchedule,invocation_run]
    exact congrArg (fun f=>(KernelIO.invocation g r Native.scalar Native.dots idx).run s >>= f) (funext ih)

theorem schedule_safe (g : Raw.Geometry) (r : Regions)
    (jobs : List (GridIndex (Host.grid g.arguments))) (bounds : RegionName → Nat) (s : State Native.Expr) :
    ScheduleSafe bounds (jobs.map (KernelIO.invocation g r Native.scalar Native.dots)) s ↔
      ScheduleSafe bounds (jobs.map (candidateInvocation g r)) s := by
  induction jobs generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    simp only [List.map_cons,ScheduleSafe,invocation_run]
    exact and_congr (invocation_safe g r idx bounds s)
      (forall_congr' (fun t=>imp_congr_right (fun _=>ih t)))

theorem equivalent (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    Equivalent (original g r extent ops) (indexShared g r extent ops) := by
  intro xs domain bounds backed initial binding
  obtain ⟨safe,final,ran,result,frame⟩:=NativeIO.public_observes g r extent ops xs domain bounds backed initial binding
  refine ⟨safe,?_,final,final,ran,?_,fun _ _ _=>rfl,frame,frame⟩
  · exact (schedule_safe g r (KernelIO.order g r extent xs) bounds initial).mp safe
  · exact (schedule_run g r (KernelIO.order g r extent xs) initial).trans ran

#axiomsClean invocation_run
#axiomsClean invocation_safe
#axiomsClean schedule_run
#axiomsClean schedule_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PairIO
