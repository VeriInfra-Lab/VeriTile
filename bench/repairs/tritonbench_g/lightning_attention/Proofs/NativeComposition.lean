import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeTerms
import bench.repairs.tritonbench_g.lightning_attention.Proofs.Frame
namespace VeriTile.Bench.Repairs.LightningAttention.Native
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section
set_option maxHeartbeats 6000000

abbrev WordMemory := RegionName → Nat → Word

def embed (m : WordMemory) : Memory Expr := fun r o => .mk .real (.bits (m r o))
def observe (p : Instructions) (plan : Plan) (m : Memory Expr) : WordMemory :=
  fun r o => (lower plan ((m r o).read .real)).eval p

@[simp] theorem observe_embed (p : Instructions) (plan : Plan) (m : WordMemory) :
    observe p plan (embed m) = m := by
  funext r o
  simp [observe,embed,Cell.read_same,lower,Graph.eval]

/-- Source-model stage graph, formed from the actual current word memory.
Every source load/store, nested scan and live gradient seed comes from the
already proved stage model. Only expression lowering is compiler specific. -/
def stageReference (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : WordMemory) : WordMemory :=
  observe p plan (stageMemory algebra dots stage d pid (embed m))

/-- Execute the complete typed source stage over expression syntax, then
lower its actual stored values to the fixed-instruction graph. -/
def stageExecute (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : WordMemory) : Option WordMemory := do
  let t ← FP.Structural.exec (engine (withDot algebra dots)) (stageKernel stage d)
    (launchState stage d pid (embed m))
  pure (observe p plan t.mem)

theorem stage_execution (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : WordMemory) :
    stageExecute p plan stage d pid m = some (stageReference p plan stage d pid m) := by
  obtain ⟨t,he,hm⟩ := stageProgram algebra dots stage d pid (embed m)
  simp only [stageExecute,he,bind,Option.bind_some,hm,Option.pure_def,stageReference]

def gridReference (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : WordMemory) : WordMemory :=
  schedule.foldl (fun mem pid => stageReference p plan stage d pid mem) m

def gridExecute (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions) :
    List PID → WordMemory → Option WordMemory
  | [],m => some m
  | pid::rest,m => do
    let next ← stageExecute p plan stage d pid m
    gridExecute p plan stage d rest next

theorem grid_execution (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : WordMemory) :
    gridExecute p plan stage d schedule m = some (gridReference p plan stage d schedule m) := by
  induction schedule generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [gridExecute,stage_execution,bind,Option.bind_some,gridReference,List.foldl_cons]
    exact ih _

def reference (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) : WordMemory :=
  gridReference p plan .inter d sb
    (gridReference p plan .intra d si (gridReference p plan .forward d sf m))

def execute (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) : Option WordMemory := do
  let f ← gridExecute p plan .forward d sf m
  let i ← gridExecute p plan .intra d si f
  gridExecute p plan .inter d sb i

/-- The current-memory three-stage execution succeeds and returns the full
instruction graph, with no successful-execution or returned-value premise. -/
theorem whole_execution (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) :
    execute p plan d sf si sb m = some (reference p plan d sf si sb m) := by
  simp only [execute,grid_execution,bind,Option.bind_some,reference]

theorem stage_frame (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (pid : PID) (m : WordMemory) (r : RegionName) (o : Nat)
    (outside : ¬Footprint stage d pid r o) : stageReference p plan stage d pid m r o = m r o := by
  simp only [stageReference,observe]
  rw [LightningAttention.stage_frame algebra dots stage d pid (embed m) r o outside]
  simp [embed,lower,Graph.eval]

theorem grid_frame (p : Instructions) (plan : Plan) (stage : Stage) (d : Host.Dimensions)
    (schedule : List PID) (m : WordMemory) (r : RegionName) (o : Nat)
    (outside : ∀ pid∈schedule, ¬Footprint stage d pid r o) :
    gridReference p plan stage d schedule m r o = m r o := by
  induction schedule generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    change gridReference p plan stage d rest (stageReference p plan stage d pid m) r o = _
    rw [ih _ (fun id hid => outside id (by simp [hid]))]
    exact stage_frame _ _ _ _ _ _ _ _ (outside pid (by simp))

def Written (d : Host.Dimensions) (sf si sb : List PID) (r : RegionName) (o : Nat) : Prop :=
  (∃ pid∈sf, Footprint .forward d pid r o) ∨
  (∃ pid∈si, Footprint .intra d pid r o) ∨
  (∃ pid∈sb, Footprint .inter d pid r o)

theorem whole_frame (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) (r : RegionName) (o : Nat)
    (outside : ¬Written d sf si sb r o) : reference p plan d sf si sb m r o = m r o := by
  unfold reference
  rw [grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside (Or.inr (Or.inr ⟨pid,hp,hw⟩))),
    grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside (Or.inr (Or.inl ⟨pid,hp,hw⟩))),
    grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside (Or.inl ⟨pid,hp,hw⟩))]

theorem whole_inputs (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) (r : RegionName) (o : Nat)
    (hr : r∈(["query","key","value","incoming"] : List RegionName)) :
    reference p plan d sf si sb m r o = m r o := by
  apply whole_frame
  rintro (⟨pid,hp,hw⟩ | ⟨pid,hp,hw⟩ | ⟨pid,hp,hw⟩) <;>
    exact input_not_written _ _ _ _ _ hr hw

/-- Only scalar-add and the two bounded ISA families transfer the complete
native graph. No variable-size dot or stage-result numerical field occurs. -/
theorem whole_native (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (sf si sb : List PID) (m : WordMemory) (numeric : LocalConformance actual selected) :
    execute actual plan d sf si sb m = some (reference selected plan d sf si sb m) ∧
    (∀ r o, ¬Written d sf si sb r o → reference selected plan d sf si sb m r o = m r o) := by
  rw [instructions_eq actual selected numeric]
  exact ⟨whole_execution _ _ _ _ _ _ _,whole_frame _ _ _ _ _ _ _⟩

structure Runtime where
  launch : Stage → WordMemory → Option WordMemory

/-- Ordinary selected-compiler binding at each complete stage. It fixes the
source transcription, layout, immediate fusion and static seed/backend plan.
The right side uses actual ISA functions, never selected-reference functions
or mathematical/whole-output equality. Only six finite profiles are observed. -/
structure Binding (actual : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : Runtime) : Prop where
  launch : ∀ stage m, runtime.launch stage m =
    some (gridReference actual plan stage d (grid stage d) m)

def runtimeRun (runtime : Runtime) (m : WordMemory) : Option WordMemory := do
  let f ← runtime.launch .forward m
  let i ← runtime.launch .intra f
  runtime.launch .inter i

theorem whole_runtime (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : Runtime) (m : WordMemory) (binding : Binding actual plan d runtime)
    (numeric : LocalConformance actual selected) :
    runtimeRun runtime m = some (reference selected plan d (grid .forward d)
      (grid .intra d) (grid .inter d) m) := by
  simp only [runtimeRun,binding.launch,bind,Option.bind_some]
  rw [instructions_eq actual selected numeric]
  rfl

#axiomsClean stage_execution
#axiomsClean grid_execution
#axiomsClean whole_execution
#axiomsClean whole_frame
#axiomsClean whole_inputs
#axiomsClean whole_native
#axiomsClean whole_runtime
#specNonCircular reference avoiding [stageExecute,gridExecute,execute,runtimeRun]
#specNonCircular Binding avoiding [whole_runtime,LocalConformance]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.Native
