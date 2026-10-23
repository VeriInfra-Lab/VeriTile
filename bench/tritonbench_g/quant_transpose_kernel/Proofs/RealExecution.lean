import bench.tritonbench_g.quantize_global.Proofs.RealHost
import bench.tritonbench_g.quant_transpose_kernel.Proofs.GroupedGrid

namespace QuantTransposeReal
open QuantTransposeFinite (Input Allocations Config Parameters parameters)
open QuantizeGlobalReal (Cell Memory Write Issue readInverse emit store applyWrites Outside
  outside_refl writes_outside writes_append finalCell writes_final_cell writes_frame emit_indices)
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 64000

/-- Shared mathematical memory uses offsets relative to the actual input view.
Physical base addresses are deliberately not interpreted as Real byte storage. -/
def inputView (a : Input) : QuantizeGlobalFinite.Input :=
  ⟨a.shape,a.strides,a.logicalOffsets,a.isCuda,a.storageOffset,a.storageLength,a.deviceIndex⟩
def allocations (a : Allocations) : QuantizeGlobalFinite.Allocations :=
  ⟨a.maximum,a.scale,a.output,a.initialOutput,a.currentDevice⟩
def readInput (a : Input) (m : Memory) (x : String) (i : Nat) : Except Issue ℝ :=
  QuantizeGlobalReal.readInput (inputView a) m x i

def sourceOutputOffset (a : Input) (pid lane : Nat) : Nat :=
  QuantTransposeFinite.row a.M a.N pid lane*1+
    QuantTransposeFinite.col a.M a.N pid lane*max a.M 1

theorem output_rematerialization (a : Input) (alloc : Allocations) (x : String)
    (pid lane : Nat) (active : QuantTransposeFinite.active (parameters a alloc x) pid lane) : sourceOutputOffset a pid lane=
      QuantTransposeFinite.offset (parameters a alloc x) pid lane := by
  have hm : 1≤a.M := by have row := active.1; dsimp [parameters] at row; omega
  simp [sourceOutputOffset,QuantTransposeFinite.offset,parameters,Nat.add_comm,Nat.max_eq_left hm]

def sourceReadLanes (a : Input) (alloc : Allocations) (m : Memory) (x : String)
    (pid : Nat) : List Nat → Except Issue (List (Nat × ℝ))
  | [] => .ok []
  | lane::rest =>
    if QuantTransposeFinite.active (parameters a alloc x) pid lane then do
      let v ← readInput a m x (QuantTransposeFinite.laneInput (parameters a alloc x) pid lane)
      let tail ← sourceReadLanes a alloc m x pid rest
      pure ((sourceOutputOffset a pid lane,v)::tail)
    else sourceReadLanes a alloc m x pid rest

/-- Independent result indexing decodes the transposed output index to its
actual strided input offset. It does not use source lanes or source execution. -/
def readOutputs (a : Input) (alloc : Allocations) (m : Memory) (x : String) :
    List Nat → Except Issue (List (Nat × ℝ))
  | [] => .ok []
  | i::rest => do
    let v ← readInput a m x (QuantTransposeFinite.inputOffset (parameters a alloc x) i)
    let tail ← readOutputs a alloc m x rest
    pure ((i,v)::tail)

theorem read_lanes_correspondence (a : Input) (alloc : Allocations) (m : Memory)
    (x : String) (pid : Nat) (lanes : List Nat) :
    sourceReadLanes a alloc m x pid lanes=
      readOutputs a alloc m x (QuantTransposeFinite.laneIndices (parameters a alloc x) pid lanes) := by
  induction lanes with
  | nil => rfl
  | cons lane rest ih =>
    by_cases ha : QuantTransposeFinite.active (parameters a alloc x) pid lane
    · have index := (QuantTransposeFinite.active_index _ _ _ ha).2
      simp [sourceReadLanes,QuantTransposeFinite.laneIndices,ha,readOutputs,index,ih,
        output_rematerialization a alloc x pid lane ha]
    · simp [sourceReadLanes,QuantTransposeFinite.laneIndices,ha,ih]

/-- All active strided input loads precede the unmasked inverse load, then
source llrint64/narrowing values precede stores. Duplicate configs stay distinct. -/
def sourceProgramPlan (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (pid : Nat) (m : Memory) : Except Issue (List Write) := do
  let loaded ← sourceReadLanes a alloc m x pid (List.range (c.blockM*c.blockN))
  let inverse ← readInverse m alloc.scale
  emit (fun v => QuantizationReal.source.eval v inverse) loaded

def programPlan (a : Input) (alloc : Allocations) (x : String) (_c : Config)
    (pid : Nat) (snapshot : Memory) : Except Issue (List Write) := do
  let loaded ← readOutputs a alloc snapshot x
    (QuantTransposeFinite.laneIndices (parameters a alloc x) pid (List.range (128*128)))
  let inverse ← readInverse snapshot alloc.scale
  emit (fun v => QuantizationReal.result v inverse) loaded

theorem source_program_correspondence (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (pid : Nat) (m : Memory) :
    sourceProgramPlan a alloc x c pid m=programPlan a alloc x c pid m := by
  simp only [sourceProgramPlan,programPlan,Config.blockM,Config.blockN,
    read_lanes_correspondence,QuantizationReal.source_scalar]

theorem read_outputs_frame (a : Input) (alloc : Allocations) (x output : String)
    (left right : Memory) (different : x≠output) (same : Outside output left right)
    (indices : List Nat) : readOutputs a alloc left x indices=readOutputs a alloc right x indices := by
  have cells : left x=right x := funext (fun i => same x i different)
  induction indices with
  | nil => rfl
  | cons i is ih => simp only [readOutputs,readInput,QuantizeGlobalReal.readInput,cells,ih]

theorem program_plan_frame (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (pid : Nat) (left right : Memory) (fresh : alloc.Fresh x)
    (same : Outside alloc.output left right) :
    programPlan a alloc x c pid left=programPlan a alloc x c pid right := by
  have read := read_outputs_frame a alloc x alloc.output left right fresh.2.2.1 same
  have scale := same alloc.scale 0 fresh.2.2.2.2.2
  simp only [programPlan,read,readInverse,scale]

def gridPlan (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (snapshot : Memory) : List Nat → Except Issue (List Write)
  | [] => .ok []
  | pid::rest => do
    let head ← programPlan a alloc x c pid snapshot
    let tail ← gridPlan a alloc x c snapshot rest
    pure (head++tail)

def executeGrid (a : Input) (alloc : Allocations) (x : String) (c : Config) :
    List Nat → Memory → Except Issue Memory
  | [],m => .ok m
  | pid::rest,m => do
    let writes ← sourceProgramPlan a alloc x c pid m
    executeGrid a alloc x c rest (applyWrites alloc.output writes m)

theorem grid_correspondence (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (snapshot : Memory) (pids : List Nat) (fresh : alloc.Fresh x)
    (m : Memory) (same : Outside alloc.output m snapshot) :
    executeGrid a alloc x c pids m =
      (gridPlan a alloc x c snapshot pids).map (fun ws => applyWrites alloc.output ws m) := by
  induction pids generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    rw [executeGrid,source_program_correspondence,
      program_plan_frame a alloc x c pid m snapshot fresh same]
    cases hp : programPlan a alloc x c pid snapshot with
    | error issue => simp [gridPlan,hp]
    | ok head =>
      simp only [bind,Except.bind]
      have frame : Outside alloc.output (applyWrites alloc.output head m) snapshot :=
        fun r i hr => (writes_outside alloc.output head m r i hr).trans (same r i hr)
      rw [ih _ frame]
      cases ht : gridPlan a alloc x c snapshot rest <;>
        simp [gridPlan,hp,ht,Except.map,writes_append]

def launchPlan (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (snapshot : Memory) : Except Issue (List Write) :=
  gridPlan a alloc x c snapshot (List.range (QuantTransposeFinite.programs (parameters a alloc x) c))

def executeLaunch (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (m : Memory) : Except Issue Memory :=
  executeGrid a alloc x c (List.range (QuantTransposeFinite.programs (parameters a alloc x) c)) m

def trialsPlan (a : Input) (alloc : Allocations) (x : String) (snapshot : Memory) :
    List Config → Except Issue (List Write)
  | [] => .ok []
  | c::cs => do
    let head ← launchPlan a alloc x c snapshot
    let tail ← trialsPlan a alloc x snapshot cs
    pure (head++tail)

def executeTrials (a : Input) (alloc : Allocations) (x : String) :
    List Config → Memory → Except Issue Memory
  | [],m => .ok m
  | c::cs,m => do
    let next ← executeLaunch a alloc x c m
    executeTrials a alloc x cs next

theorem trials_correspondence (a : Input) (alloc : Allocations) (x : String)
    (snapshot : Memory) (configs : List Config) (fresh : alloc.Fresh x)
    (m : Memory) (same : Outside alloc.output m snapshot) :
    executeTrials a alloc x configs m =
      (trialsPlan a alloc x snapshot configs).map (fun ws => applyWrites alloc.output ws m) := by
  induction configs generalizing m with
  | nil => rfl
  | cons c cs ih =>
    rw [executeTrials,executeLaunch,grid_correspondence a alloc x c snapshot _ fresh m same]
    change ((launchPlan a alloc x c snapshot).map fun ws => applyWrites alloc.output ws m).bind _ = _
    cases hp : launchPlan a alloc x c snapshot with
    | error issue => simp [trialsPlan,hp,Except.map,Except.bind]
    | ok head =>
      simp only [Except.map,Except.bind]
      have frame : Outside alloc.output (applyWrites alloc.output head m) snapshot :=
        fun r i hr => (writes_outside alloc.output head m r i hr).trans (same r i hr)
      rw [ih _ frame]
      cases ht : trialsPlan a alloc x snapshot cs <;>
        simp [trialsPlan,hp,ht,Except.map,writes_append]

theorem read_outputs_indices (a : Input) (alloc : Allocations) (m : Memory) (x : String)
    (indices : List Nat) (values : List (Nat × ℝ))
    (h : readOutputs a alloc m x indices=.ok values) : values.map Prod.fst=indices := by
  induction indices generalizing values with
  | nil => simp [readOutputs] at h; subst values; rfl
  | cons i is ih =>
    cases hv : readInput a m x (QuantTransposeFinite.inputOffset (parameters a alloc x) i) with
    | error issue => simp [readOutputs,hv] at h
    | ok v =>
      cases ht : readOutputs a alloc m x is with
      | error issue => simp [readOutputs,hv,ht] at h
      | ok tail =>
        simp [readOutputs,hv,ht] at h
        subst values
        simp [ih tail ht]

theorem program_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (pid : Nat) (snapshot : Memory) (writes : List Write)
    (h : programPlan a alloc x c pid snapshot=.ok writes) :
    writes.map Prod.fst=QuantTransposeFinite.laneIndices (parameters a alloc x) pid (List.range (128*128)) := by
  cases hl : readOutputs a alloc snapshot x
      (QuantTransposeFinite.laneIndices (parameters a alloc x) pid (List.range (128*128))) with
  | error issue => simp [programPlan,hl] at h
  | ok loaded =>
    cases hs : readInverse snapshot alloc.scale with
    | error issue => simp [programPlan,hl,hs] at h
    | ok inverse =>
      simp [programPlan,hl,hs] at h
      exact (emit_indices _ loaded writes h).trans (read_outputs_indices _ _ _ _ _ _ hl)

theorem grid_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (snapshot : Memory) (pids : List Nat) (writes : List Write)
    (h : gridPlan a alloc x c snapshot pids=.ok writes) :
    writes.map Prod.fst=QuantTransposeFinite.footprint (parameters a alloc x) pids := by
  induction pids generalizing writes with
  | nil => simp [gridPlan] at h; subst writes; rfl
  | cons pid pids ih =>
    cases hh : programPlan a alloc x c pid snapshot with
    | error issue => simp [gridPlan,hh] at h
    | ok head =>
      cases ht : gridPlan a alloc x c snapshot pids with
      | error issue => simp [gridPlan,hh,ht] at h
      | ok tail =>
        simp [gridPlan,hh,ht] at h
        subst writes
        simp [QuantTransposeFinite.footprint,program_plan_indices _ _ _ _ _ _ _ hh,ih tail ht]

def trialIndices (p : Parameters) (configs : List Config) : List Nat :=
  configs.flatMap (fun c => QuantTransposeFinite.footprint p (List.range (QuantTransposeFinite.programs p c)))

theorem trials_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (snapshot : Memory) (configs : List Config) (writes : List Write)
    (h : trialsPlan a alloc x snapshot configs=.ok writes) :
    writes.map Prod.fst=trialIndices (parameters a alloc x) configs := by
  induction configs generalizing writes with
  | nil => simp [trialsPlan] at h; subst writes; rfl
  | cons c cs ih =>
    cases hh : launchPlan a alloc x c snapshot with
    | error issue => simp [trialsPlan,hh] at h
    | ok head =>
      cases ht : trialsPlan a alloc x snapshot cs with
      | error issue => simp [trialsPlan,hh,ht] at h
      | ok tail =>
        simp [trialsPlan,hh,ht] at h
        subst writes
        simp [trialIndices,grid_plan_indices _ _ _ _ _ _ _ hh,ih tail ht]

theorem complete_trials_index (p : Parameters) (trials : List Config) (last : Config)
    (i : Nat) (hi : i<p.n) : i∈trialIndices p (trials++[last]) := by
  apply List.mem_flatMap.mpr
  exact ⟨last,by simp,QuantTransposeFinite.footprint_complete p last i hi⟩

theorem trial_indices_bound (p : Parameters) (configs : List Config) (i : Nat)
    (hi : i∈trialIndices p configs) : i<p.n := by
  obtain ⟨c,hc,hgrid⟩ := List.mem_flatMap.mp hi
  obtain ⟨pid,hpid,hlanes⟩ := List.mem_flatMap.mp hgrid
  obtain ⟨lane,hlane,hindex⟩ := List.mem_filterMap.mp hlanes
  split_ifs at hindex with active
  · have eqi : QuantTransposeFinite.offset p pid lane=i := Option.some.inj hindex
    rw [←eqi]
    exact (QuantTransposeFinite.active_index _ _ _ active).1

/-- Every actually launched group has a positive divisor. Totalized Nat
division outside the launched grid is never needed by wrapper execution. -/
theorem launched_group_defined (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (pid : Nat) (hp : pid<QuantTransposeFinite.programs (parameters a alloc x) c) :
    0<QuantTransposeFinite.groupSize (QuantTransposeFinite.gridM a.M) (QuantTransposeFinite.gridN a.N) pid := by
  have bound : pid<QuantTransposeFinite.gridM a.M*QuantTransposeFinite.gridN a.N := hp
  have hn : 0<QuantTransposeFinite.gridN a.N := by nlinarith
  exact (QuantTransposeFinite.tile_bounds _ _ pid hn bound).1

#specNonCircular readOutputs avoiding [sourceReadLanes,sourceProgramPlan,executeGrid,executeLaunch,executeTrials]
#specNonCircular programPlan avoiding [sourceReadLanes,sourceProgramPlan,executeGrid,executeLaunch,executeTrials]
#specNonCircular trialsPlan avoiding [sourceReadLanes,sourceProgramPlan,executeGrid,executeLaunch,executeTrials,applyWrites]
#axiomsClean output_rematerialization
#axiomsClean read_lanes_correspondence
#axiomsClean source_program_correspondence
#axiomsClean read_outputs_frame
#axiomsClean program_plan_frame
#axiomsClean grid_correspondence
#axiomsClean trials_correspondence
#axiomsClean read_outputs_indices
#axiomsClean program_plan_indices
#axiomsClean grid_plan_indices
#axiomsClean trials_plan_indices
#axiomsClean complete_trials_index
#axiomsClean trial_indices_bound
#axiomsClean launched_group_defined
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeReal
