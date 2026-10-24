import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeGrid

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeHost
open VeriTile.Triton FP.Structural WholeSource WholeGrid
open Typed (Parameters)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Exactly the source's 3*4*2*2 autotune configurations. -/
structure Config where
  blockColumns : Nat
  blockRows : Nat
  stages : Nat
  warps : Nat
  columns_mem : blockColumns ∈ [32,64,128]
  rows_mem : blockRows ∈ [1,2,4,8]
  stages_mem : stages ∈ [3,4]
  warps_mem : warps ∈ [4,8]

theorem Config.rows_pos (c : Config) : 0<c.blockRows := by
  have h:=c.rows_mem
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

theorem Config.columns_pos (c : Config) : 0<c.blockColumns := by
  have h:=c.columns_mem
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

/-- FP32 tensor views. Bases and strides are element addresses; the finite
adapter maps these to aligned four-byte accesses. Zero strides and aliasing
between the two input views are allowed. Rank is fixed by this host branch. -/
structure Inputs where
  rows : Nat
  columns : Nat
  vectorLength : Nat
  matrixBase : Nat
  vectorBase : Nat
  strideRow : Nat
  strideColumn : Nat
  strideVector : Nat
  device : Nat

def Inputs.matrixAddress (x : Inputs) (i j : Nat) := x.matrixBase+i*x.strideRow+j*x.strideColumn
def Inputs.vectorAddress (x : Inputs) (j : Nat) := x.vectorBase+j*x.strideVector

def Inputs.parameters (x : Inputs) (c : Config) : Parameters where
  a:="A"; b:="B"; out:="C"
  rows:=x.rows; columns:=x.columns
  strideRow:=x.strideRow; strideColumn:=x.strideColumn; strideVector:=x.strideVector
  strideOut:=1; blockRows:=c.blockRows; blockColumns:=c.blockColumns

/-- Ordinary fresh torch.empty: arbitrary initial output contents, exact
one-dimensional extent, and preservation of every word outside that extent. -/
structure Allocated (α : Type) (x : Inputs) (entry : Nat → α) (preexisting : Nat → Prop) where
  base : Nat
  memory : Nat → α
  fresh : ∀ i,i<x.rows → ¬preexisting (base+i)
  matrixExisting : ∀ i j,i<x.rows → j<x.columns → preexisting (x.matrixAddress i j)
  vectorExisting : ∀ j,j<x.vectorLength → preexisting (x.vectorAddress j)
  frame : ∀ addr,(∀ i,i<x.rows → addr≠base+i) → memory addr=entry addr

variable {α : Type} [Inhabited α] {x : Inputs} {entry : Nat → α} {preexisting : Nat → Prop}

theorem Allocated.protected_frame (a : Allocated α x entry preexisting)
    (addr : Nat) (old : preexisting addr) : a.memory addr=entry addr := by
  apply a.frame
  intro i hi eq
  apply a.fresh i hi
  simpa only [←eq] using old

def Allocated.state (a : Allocated α x entry preexisting) : State α where
  mem:=fun region offset => .mk .fp32 (a.memory
    (if region="A" then x.matrixBase+offset else if region="B" then x.vectorBase+offset else a.base+offset))
  regs:=fun _ _ _ => none
  pids:=fun _ => 0
  numPids:=fun _ => 1
  undef:=fun d _ _ => (Cell.mk .fp32 default).read d

/-- All output-region addresses are committed, including the exterior. The
source frame, rather than a truncating commit, establishes physical safety. -/
def Allocated.commit (a : Allocated α x entry preexisting) (readOut : Nat → α) : Nat → α :=
  fun addr => if a.base≤addr then readOut (addr-a.base) else a.memory addr

@[simp] theorem Allocated.commit_value (a : Allocated α x entry preexisting) (readOut : Nat → α) (i : Nat) :
    a.commit readOut (a.base+i)=readOut i := by simp [Allocated.commit]

theorem Allocated.commit_frame (a : Allocated α x entry preexisting) (readOut : Nat → α)
    (frame : ∀ i,x.rows ≤ i → readOut i=a.memory (a.base+i))
    (addr : Nat) (outside : ∀ i,i<x.rows → addr≠a.base+i) :
    a.commit readOut addr=a.memory addr := by
  unfold Allocated.commit
  split
  · rename_i hb
    have hi : x.rows≤addr-a.base := by
      by_contra h
      exact outside (addr-a.base) (by omega) (by omega)
    rw [frame _ hi]
    congr 1
    omega
  · rfl

structure Trial (x : Inputs) where
  config : Config
  order : List Nat
  schedule : order.Perm (List.range (gridSize (x.parameters config)))

def trialState (x : Inputs) (c : Config) (s : State α) : State α :=
  {s with numPids:=fun ax => if ax=0 then gridSize (x.parameters c) else 1}

def runTrial (family : Config → Nat → Algebra α) (t : Trial x) (s : State α) : Option (State α) :=
  runTiles (family t.config) (x.parameters t.config) t.order (trialState x t.config s)

def runTrials (family : Config → Nat → Algebra α) : List (Trial x) → State α → Option (State α)
  | [],s => some s
  | t::rest,s => (runTrial family t s).bind (runTrials family rest)

/-- Repeated autotune launches are complete original source executions. No
numerical result, timing value or selected configuration is assumed. -/
theorem trials_result (family : Config → Nat → Algebra α) (trials : List (Trial x)) (s : State α) :
    ∃ final,runTrials family trials s=some final ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ x.rows ≤ offset) → final.mem region offset=s.mem region offset) := by
  induction trials generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons t rest ih =>
    obtain ⟨middle,ran,_values,frame⟩ := grid_result (family t.config) (x.parameters t.config)
      t.config.rows_pos t.config.columns_pos rfl (by change ("A" : RegionName)≠"C"; decide) (by change ("B" : RegionName)≠"C"; decide)
      t.order t.schedule (trialState x t.config s)
    obtain ⟨final,tailRan,tailFrame⟩ := ih middle
    refine ⟨final,?_,?_⟩
    · change (runTrial family t s).bind _=_
      rw [show runTrial family t s=some middle from ran]
      exact tailRan
    · intro r o exterior
      rw [tailFrame r o exterior]
      exact frame r o exterior

structure Plan (x : Inputs) where
  trials : List (Trial x)
  selected : Trial x

def runPlan (family : Config → Nat → Algebra α) (plan : Plan x) (s : State α) : Option (State α) := do
  let middle ← runTrials family plan.trials s
  runTrial family plan.selected middle

/-- Autotuning and the final selected launch preserve every input and exterior
cell, and the final output is evaluated from the entry source memory. -/
theorem plan_result (family : Config → Nat → Algebra α) (plan : Plan x) (s : State α) :
    ∃ final,runPlan family plan s=some final ∧
      (∀ i,i<x.rows → final.mem "C" i=.mk .fp32
        (rowValue (family plan.selected.config) (x.parameters plan.selected.config)
          plan.selected.config.rows_pos s.mem i)) ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ x.rows ≤ offset) → final.mem region offset=s.mem region offset) := by
  obtain ⟨middle,trialRan,trialFrame⟩ := trials_result family plan.trials s
  obtain ⟨final,ran,values,frame⟩ := grid_result (family plan.selected.config)
    (x.parameters plan.selected.config) plan.selected.config.rows_pos plan.selected.config.columns_pos
    rfl (by change ("A" : RegionName)≠"C"; decide) (by change ("B" : RegionName)≠"C"; decide) plan.selected.order plan.selected.schedule
    (trialState x plan.selected.config middle)
  refine ⟨final,?_,?_,?_⟩
  · simp only [runPlan,trialRan,Option.bind_some]
    exact ran
  · intro i hi
    change final.mem (x.parameters plan.selected.config).out i=_
    rw [values i hi]
    congr 1
    apply tileValue_congr
    · intro j
      change (middle.mem "A" j).read .fp32=(s.mem "A" j).read .fp32
      rw [trialFrame "A" j (Or.inl (by decide))]
    · intro j
      change (middle.mem "B" j).read .fp32=(s.mem "B" j).read .fp32
      rw [trialFrame "B" j (Or.inl (by decide))]
  · intro region offset exterior
    rw [frame region offset exterior]
    exact trialFrame region offset exterior

structure Output (α : Type) where
  length : Nat
  dtype : FloatDType
  device : Nat
  stride : Nat
  base : Nat
  memory : Nat → α

def makeOutput (a : Allocated α x entry preexisting) (final : State α) : Output α :=
  ⟨x.rows,.fp32,x.device,1,a.base,a.commit (fun i => (final.mem "C" i).read .fp32)⟩

structure Runtime (α : Type) (x : Inputs) (entry : Nat → α) (preexisting : Nat → Prop) where
  allocate : Option (Allocated α x entry preexisting)
  launchReady : (a : Allocated α x entry preexisting) → Bool
  plan : (a : Allocated α x entry preexisting) → Plan x

inductive HostOutcome (α : Type) where
  | incompatibleDimensions
  | allocationFailure
  | setupFailure
  | sourceFailure
  | returned (out : Output α)

def execute (family : Config → Nat → Algebra α) (rt : Runtime α x entry preexisting) : HostOutcome α :=
  if x.columns≠x.vectorLength then .incompatibleDimensions else
    match rt.allocate with
    | none => .allocationFailure
    | some a => if rt.launchReady a then
        match runPlan family (rt.plan a) a.state with
        | none => .sourceFailure
        | some final => .returned (makeOutput a final)
      else .setupFailure

theorem execute_mismatch (family : Config → Nat → Algebra α) (rt : Runtime α x entry preexisting)
    (mismatch : x.columns≠x.vectorLength) : execute family rt=.incompatibleDimensions := by
  simp [execute,mismatch]

/-- The ordinary runtime contract supplies allocation, setup and schedules,
never kernel values. Successful setup reaches the original return with exact
source values, shape/dtype/device/stride and physical exterior preservation. -/
theorem execute_returns (family : Config → Nat → Algebra α) (rt : Runtime α x entry preexisting)
    (compatible : x.columns=x.vectorLength) (a : Allocated α x entry preexisting)
    (allocated : rt.allocate=some a) (ready : rt.launchReady a=Bool.true) :
    ∃ out,execute family rt=.returned out ∧ out.length=x.rows ∧ out.dtype=.fp32 ∧
      out.device=x.device ∧ out.stride=1 ∧ out.base=a.base ∧
      (∀ i,i<x.rows → out.memory (out.base+i)=
        rowValue (family (rt.plan a).selected.config) (x.parameters (rt.plan a).selected.config)
          (rt.plan a).selected.config.rows_pos a.state.mem i) ∧
      (∀ addr,(∀ i,i<x.rows → addr≠out.base+i) → out.memory addr=a.memory addr) ∧
      (∀ addr,preexisting addr → out.memory addr=entry addr) := by
  obtain ⟨final,ran,values,frame⟩ := plan_result family (rt.plan a) a.state
  have exterior (i : Nat) (hi : x.rows ≤ i) :
      (final.mem "C" i).read .fp32=a.memory (a.base+i) := by
    rw [frame "C" i (Or.inr hi)]
    rfl
  have physicalFrame := a.commit_frame _ exterior
  refine ⟨makeOutput a final,?_,rfl,rfl,rfl,rfl,rfl,?_,physicalFrame,?_⟩
  · simp [execute,compatible,allocated,ready,ran]
  · intro i hi
    simp only [makeOutput]
    rw [a.commit_value,values i hi]
    rfl
  · intro addr hp
    simp only [makeOutput]
    rw [physicalFrame addr (by
      intro i hi same
      apply a.fresh i hi
      simpa only [←same] using hp)]
    exact a.protected_frame addr hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeHost
