import bench.tritonbench_g.quantize_global.Proofs.RealScalar
import bench.tritonbench_g.quantize_global.Kernels
import bench.tritonbench_g.quantize_global.Proofs.SourceIndices

local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace QuantizeGlobalReal
open QuantizeGlobalFinite (Input Allocations Config)
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000

@[simp] theorem except_bind_ok {ε α β : Type} (v : α) (f : α → Except ε β) :
    (Except.ok v >>= f)=f v := rfl
@[simp] theorem except_bind_error {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (Except.error e >>= f)=Except.error e := rfl
@[simp] theorem except_map_ok {ε α β : Type} (f : α → β) (v : α) :
    (Except.ok v : Except ε α).map f=Except.ok (f v) := rfl
@[simp] theorem except_map_error {ε α β : Type} (f : α → β) (e : ε) :
    (Except.error e : Except ε α).map f=Except.error e := rfl

inductive Cell where
  | real (value : ℝ)
  | scale (value : Option ℝ)
  | byte (value : BitVec 8)
abbrev Memory := String → Nat → Option Cell
abbrev Write := Nat × BitVec 8

inductive Issue where
  | inputAccess (index : Nat)
  | inputMissing (index : Nat)
  | inverseMissing
  | conversionUndefined (index : Nat)
  deriving DecidableEq, Repr

def readInput (a : Input) (m : Memory) (x : String) (i : Nat) : Except Issue ℝ :=
  if a.storageOffset+i<a.storageLength then
    match m x i with
    | some (.real v) => .ok v
    | _ => .error (.inputMissing i)
  else .error (.inputAccess i)

def readInputs (a : Input) (m : Memory) (x : String) :
    List Nat → Except Issue (List (Nat × ℝ))
  | [] => .ok []
  | i::is => do
    let v ← readInput a m x i
    let rest ← readInputs a m x is
    pure ((i,v)::rest)

def readInverse (m : Memory) (scale : String) : Except Issue (Option ℝ) :=
  match m scale 0 with
  | some (.scale v) => .ok v
  | _ => .error .inverseMissing

def emit (scalar : ℝ → Option (BitVec 8)) :
    List (Nat × ℝ) → Except Issue (List Write)
  | [] => .ok []
  | (i,v)::rest => do
    let byte ← match scalar v with
      | none => .error (.conversionUndefined i)
      | some b => .ok b
    let tail ← emit scalar rest
    pure ((i,byte)::tail)

/-- Exact mathematical indices of the source BLOCK1024/BLOCK2048 tiles. -/
def activeIndices (n : Nat) (c : Config) (pid : Nat) : List Nat :=
  ((List.range c.block).map (QuantizeGlobalFinite.offset c pid)).filter (fun i => i<n)

/-- All masked source input loads precede the unmasked inverse load, and all
scalar values precede the program's stores. An undefined scale remains a value
marker until an active llrint consumes it. -/
def sourceProgramPlan (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (pid : Nat) (m : Memory) : Except Issue (List Write) := do
  let loaded ← readInputs a m x (activeIndices a.numel c pid)
  let inverse ← readInverse m alloc.scale
  emit (fun v => QuantizationReal.source.eval v inverse) loaded

def programPlan (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (pid : Nat) (snapshot : Memory) : Except Issue (List Write) := do
  let loaded ← readInputs a snapshot x (activeIndices a.numel c pid)
  let inverse ← readInverse snapshot alloc.scale
  emit (fun v => QuantizationReal.result v inverse) loaded

theorem source_program_correspondence (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (pid : Nat) (m : Memory) :
    sourceProgramPlan a alloc x c pid m = programPlan a alloc x c pid m := by
  simp only [sourceProgramPlan,programPlan,QuantizationReal.source_scalar]

def store (output : String) (w : Write) (m : Memory) : Memory :=
  fun r i => if r=output ∧ i=w.1 then some (.byte w.2) else m r i

def applyWrites (output : String) : List Write → Memory → Memory
  | [],m => m
  | w::ws,m => applyWrites output ws (store output w m)

def Outside (output : String) (left right : Memory) : Prop :=
  ∀ r i, r≠output → left r i=right r i

theorem outside_refl (output : String) (m : Memory) : Outside output m m := fun _ _ _ => rfl

theorem writes_outside (output : String) (ws : List Write) (m : Memory) :
    Outside output (applyWrites output ws m) m := by
  induction ws generalizing m with
  | nil => exact outside_refl output m
  | cons w ws ih =>
    intro r i hr
    exact (ih (store output w m) r i hr).trans (by simp [store,hr])

theorem writes_append (output : String) (left right : List Write) (m : Memory) :
    applyWrites output (left++right) m = applyWrites output right (applyWrites output left m) := by
  induction left generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih _

theorem read_inputs_frame (a : Input) (x output : String) (left right : Memory)
    (different : x≠output) (same : Outside output left right) (indices : List Nat) :
    readInputs a left x indices = readInputs a right x indices := by
  have cells : left x=right x := funext (fun i => same x i different)
  induction indices with
  | nil => rfl
  | cons i is ih => simp only [readInputs,readInput,cells,ih]

theorem program_plan_frame (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (pid : Nat) (left right : Memory) (fresh : alloc.Fresh x)
    (same : Outside alloc.output left right) :
    programPlan a alloc x c pid left = programPlan a alloc x c pid right := by
  have read := read_inputs_frame a x alloc.output left right fresh.2.2.1 same
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
  gridPlan a alloc x c snapshot (List.range (QuantizeGlobalFinite.programs a.numel c))

def executeLaunch (a : Input) (alloc : Allocations) (x : String) (c : Config)
    (m : Memory) : Except Issue Memory :=
  executeGrid a alloc x c (List.range (QuantizeGlobalFinite.programs a.numel c)) m

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

/-- Independent last actual store formula on raw int8 cells. -/
def finalCell (output : String) (writes : List Write) (initial : Memory)
    (r : String) (i : Nat) : Option Cell :=
  if r=output then
    match writes.reverse.find? (fun w => w.1==i) with
    | none => initial r i
    | some w => some (.byte w.2)
  else initial r i

theorem writes_final_cell (output : String) (writes : List Write) (initial : Memory)
    (r : String) (i : Nat) :
    applyWrites output writes initial r i = finalCell output writes initial r i := by
  induction writes generalizing initial with
  | nil => simp [applyWrites,finalCell]
  | cons w ws ih =>
    rw [applyWrites,ih]
    by_cases hr : r=output
    · subst r
      simp only [finalCell,if_true,List.reverse_cons,List.find?_append]
      cases ht : ws.reverse.find? (fun v => v.1==i) with
      | some v => rfl
      | none =>
        by_cases he : w.1=i
        · simp [List.find?,store,he]
        · have hb : (w.1==i)=false := by simp [he]
          have hn : i≠w.1 := Ne.symm he
          simp [List.find?,store,hb,hn]
    · simp [finalCell,hr,store]

theorem writes_frame (output : String) (writes : List Write) (initial : Memory)
    (r : String) (i : Nat) (untouched : r≠output ∨ ∀ w∈writes, w.1≠i) :
    applyWrites output writes initial r i=initial r i := by
  induction writes generalizing initial with
  | nil => rfl
  | cons w ws ih =>
    have tail : r≠output ∨ ∀ v∈ws, v.1≠i := by
      rcases untouched with h|h
      · exact Or.inl h
      · exact Or.inr (fun v hv => h v (by simp [hv]))
    rw [applyWrites,ih _ tail]
    rcases untouched with hr|hi
    · simp [store,hr]
    · have hw := hi w (by simp)
      simp [store,Ne.symm hw]

theorem read_inputs_indices (a : Input) (m : Memory) (x : String)
    (indices : List Nat) (values : List (Nat × ℝ))
    (h : readInputs a m x indices=.ok values) : values.map Prod.fst=indices := by
  induction indices generalizing values with
  | nil => simp [readInputs] at h; subst values; rfl
  | cons i is ih =>
    cases hv : readInput a m x i with
    | error issue => simp [readInputs,hv] at h
    | ok v =>
      cases ht : readInputs a m x is with
      | error issue => simp [readInputs,hv,ht] at h
      | ok tail =>
        simp [readInputs,hv,ht] at h
        subst values
        simp [ih tail ht]

theorem emit_indices (scalar : ℝ → Option (BitVec 8)) (values : List (Nat × ℝ))
    (writes : List Write) (h : emit scalar values=.ok writes) :
    writes.map Prod.fst=values.map Prod.fst := by
  induction values generalizing writes with
  | nil => simp [emit] at h; subst writes; rfl
  | cons head rest ih =>
    rcases head with ⟨i,v⟩
    cases hv : scalar v with
    | none => simp [emit,hv] at h
    | some byte =>
      cases ht : emit scalar rest with
      | error issue => simp [emit,hv,ht] at h
      | ok tail =>
        simp [emit,hv,ht] at h
        subst writes
        simp [ih tail ht]

theorem program_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (pid : Nat) (snapshot : Memory) (writes : List Write)
    (h : programPlan a alloc x c pid snapshot=.ok writes) :
    writes.map Prod.fst=activeIndices a.numel c pid := by
  cases hl : readInputs a snapshot x (activeIndices a.numel c pid) with
  | error issue => simp [programPlan,hl] at h
  | ok loaded =>
    cases hs : readInverse snapshot alloc.scale with
    | error issue => simp [programPlan,hl,hs] at h
    | ok inverse =>
      simp [programPlan,hl,hs] at h
      exact (emit_indices _ loaded writes h).trans (read_inputs_indices _ _ _ _ _ hl)

def gridIndices (n : Nat) (c : Config) (pids : List Nat) : List Nat :=
  (QuantizeGlobalFinite.footprint c pids).filter (fun i => i<n)

theorem grid_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (c : Config) (snapshot : Memory) (pids : List Nat) (writes : List Write)
    (h : gridPlan a alloc x c snapshot pids=.ok writes) :
    writes.map Prod.fst=gridIndices a.numel c pids := by
  induction pids generalizing writes with
  | nil => simp [gridPlan] at h; subst writes; rfl
  | cons pid rest ih =>
    cases hp : programPlan a alloc x c pid snapshot with
    | error issue => simp [gridPlan,hp] at h
    | ok head =>
      cases ht : gridPlan a alloc x c snapshot rest with
      | error issue => simp [gridPlan,hp,ht] at h
      | ok tail =>
        simp [gridPlan,hp,ht] at h
        subst writes
        rw [List.map_append,program_plan_indices _ _ _ _ _ _ _ hp,ih tail ht]
        simp [gridIndices,QuantizeGlobalFinite.footprint,activeIndices,List.filter_append]

def trialIndices (n : Nat) (configs : List Config) : List Nat :=
  configs.flatMap fun c => gridIndices n c (List.range (QuantizeGlobalFinite.programs n c))

theorem trials_plan_indices (a : Input) (alloc : Allocations) (x : String)
    (snapshot : Memory) (configs : List Config) (writes : List Write)
    (h : trialsPlan a alloc x snapshot configs=.ok writes) :
    writes.map Prod.fst=trialIndices a.numel configs := by
  induction configs generalizing writes with
  | nil => simp [trialsPlan] at h; subst writes; rfl
  | cons c cs ih =>
    cases hp : launchPlan a alloc x c snapshot with
    | error issue => simp [trialsPlan,hp] at h
    | ok head =>
      cases ht : trialsPlan a alloc x snapshot cs with
      | error issue => simp [trialsPlan,hp,ht] at h
      | ok tail =>
        simp [trialsPlan,hp,ht] at h
        subst writes
        rw [List.map_append,grid_plan_indices _ _ _ _ _ _ _ hp,ih tail ht]
        rfl

theorem complete_grid_index (n i : Nat) (c : Config) (hi : i<n) :
    i∈gridIndices n c (List.range (QuantizeGlobalFinite.programs n c)) := by
  have cover : i∈QuantizeGlobalFinite.footprint c
      (List.range (QuantizeGlobalFinite.programs n c)) := by
    rw [QuantizeGlobalFinite.footprint_membership,List.mem_range]
    cases c <;> simp [QuantizeGlobalFinite.programs,Config.block] <;> omega
  exact List.mem_filter.mpr ⟨cover,by simpa using hi⟩

theorem complete_trials_index (n i : Nat) (trials : List Config) (last : Config)
    (hi : i<n) : i∈trialIndices n (trials++[last]) := by
  apply List.mem_flatMap.mpr
  exact ⟨last,by simp,complete_grid_index n i last hi⟩

theorem trial_indices_bound (n i : Nat) (configs : List Config)
    (hi : i∈trialIndices n configs) : i<n := by
  obtain ⟨c,_,hc⟩ := List.mem_flatMap.mp hi
  simpa using (List.mem_filter.mp hc).2

#specNonCircular programPlan avoiding [sourceProgramPlan,executeGrid,executeLaunch,executeTrials]
#specNonCircular trialsPlan avoiding [sourceProgramPlan,executeGrid,executeLaunch,executeTrials,applyWrites]
#specNonCircular finalCell avoiding [store,applyWrites,executeGrid,executeLaunch,executeTrials]
#axiomsClean source_program_correspondence
#axiomsClean writes_outside
#axiomsClean writes_append
#axiomsClean read_inputs_frame
#axiomsClean program_plan_frame
#axiomsClean grid_correspondence
#axiomsClean trials_correspondence
#axiomsClean writes_final_cell
#axiomsClean writes_frame
#axiomsClean read_inputs_indices
#axiomsClean emit_indices
#axiomsClean program_plan_indices
#axiomsClean grid_plan_indices
#axiomsClean trials_plan_indices
#axiomsClean complete_grid_index
#axiomsClean complete_trials_index
#axiomsClean trial_indices_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalReal
