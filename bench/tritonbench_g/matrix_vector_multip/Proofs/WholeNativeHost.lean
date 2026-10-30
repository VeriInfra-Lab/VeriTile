import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNative
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNativeHost
open VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost WholeNative
noncomputable section
set_option maxHeartbeats 6000000

variable {x : Inputs} {entry : Nat → BitVec 32} {preexisting : Nat → Prop}

def liftAllocated (a : Allocated (BitVec 32) x entry preexisting) :
    Allocated Payload x (fun i => .word (entry i)) preexisting where
  base:=a.base
  memory:=fun i => .word (a.memory i)
  fresh:=a.fresh
  matrixExisting:=a.matrixExisting
  vectorExisting:=a.vectorExisting
  frame:=by intro addr outside; rw [a.frame addr outside]

def family (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (pid : Nat) : Algebra Payload := algebra B (graphs c pid)

def nativeRow (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (a : Allocated (BitVec 32) x entry preexisting) (i : Nat) : BitVec 32 :=
  tileFormula B (x.parameters c) (graphs c (i/c.blockRows)) (i/c.blockRows) a.state.mem
    ⟨i%c.blockRows,Nat.mod_lt _ c.rows_pos⟩

theorem source_row (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (a : Allocated (BitVec 32) x entry preexisting) (i : Nat) :
    rowValue (family B graphs c) (x.parameters c) c.rows_pos (liftAllocated a).state.mem i=
      .word (nativeRow B graphs c a i) := by
  have lifted : (liftAllocated a).state.mem=liftMemory a.state.mem := by
    funext r i
    simp only [Allocated.state,liftAllocated,liftMemory]
    split <;> try rfl
    all_goals split <;> rfl
  unfold rowValue family nativeRow
  rw [lifted]
  exact tileFormula_eq B (x.parameters c) _ _ _ _

/-- Original entry words, independent of allocation and tuning execution. -/
def entryMemory (x : Inputs) (entry : Nat → BitVec 32) : RegionName → Nat → Cell (BitVec 32) :=
  fun region offset => .mk .fp32
    (if region="A" then entry (x.matrixBase+offset)
     else if region="B" then entry (x.vectorBase+offset) else 0)

theorem matrixLoad_entry (a : Allocated (BitVec 32) x entry preexisting) (c : Config)
    (pid off : Nat) (i : Fin c.blockRows) (j : Fin c.blockColumns) :
    matrixLoad (x.parameters c) pid off a.state.mem i j=
      matrixLoad (x.parameters c) pid off (entryMemory x entry) i j := by
  unfold matrixLoad
  split
  · rename_i active
    change a.memory (x.matrixBase+aOffset (x.parameters c) pid off i.val j.val)=
      entry (x.matrixBase+aOffset (x.parameters c) pid off i.val j.val)
    have address : x.matrixBase+aOffset (x.parameters c) pid off i.val j.val=
        x.matrixAddress (row (x.parameters c) pid i.val) (off+j.val) := by
      unfold aOffset Inputs.matrixAddress Inputs.parameters
      ring
    rw [address]
    exact a.protected_frame _ (a.matrixExisting _ _ active.1 active.2)
  · rfl

theorem vectorLoad_entry (compatible : x.columns=x.vectorLength)
    (a : Allocated (BitVec 32) x entry preexisting) (c : Config) (off : Nat) (j : Fin c.blockColumns) :
    vectorLoad (x.parameters c) off a.state.mem j=
      vectorLoad (x.parameters c) off (entryMemory x entry) j := by
  unfold vectorLoad
  split
  · rename_i active
    change a.memory (x.vectorBase+bOffset (x.parameters c) off j.val)=
      entry (x.vectorBase+bOffset (x.parameters c) off j.val)
    have address : x.vectorBase+bOffset (x.parameters c) off j.val=x.vectorAddress (off+j.val) := by
      unfold bOffset Inputs.vectorAddress Inputs.parameters
      ring
    rw [address]
    exact a.protected_frame _ (a.vectorExisting _ (by simpa only [←compatible] using active))
  · rfl

theorem accumulate_entry (B : Scalars) (c : Config) (g : Graph c.blockRows c.blockColumns)
    (compatible : x.columns=x.vectorLength) (a : Allocated (BitVec 32) x entry preexisting)
    (pid off : Nat) (acc : Acc (BitVec 32) (x.parameters c)) :
    accumulate B (x.parameters c) g pid a.state.mem off acc=
      accumulate B (x.parameters c) g pid (entryMemory x entry) off acc := by
  conv_lhs => rw [accumulate]
  conv_rhs => rw [accumulate]
  split
  · have stepEq : next B (x.parameters c) g pid off a.state.mem acc=
        next B (x.parameters c) g pid off (entryMemory x entry) acc := by
      funext i
      simp only [next,matrixLoad_entry,vectorLoad_entry compatible]
    rw [stepEq]
    exact accumulate_entry B c g compatible a pid (off+c.blockColumns) _
  · rfl
termination_by x.columns-off
decreasing_by
  simp_wf
  simp only [Inputs.parameters] at *
  omega

def nativeFormula (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (x : Inputs) (entry : Nat → BitVec 32) (i : Nat) : BitVec 32 :=
  tileFormula B (x.parameters c) (graphs c (i/c.blockRows)) (i/c.blockRows) (entryMemory x entry)
    ⟨i%c.blockRows,Nat.mod_lt _ c.rows_pos⟩

theorem nativeRow_entry (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (c : Config) (compatible : x.columns=x.vectorLength)
    (a : Allocated (BitVec 32) x entry preexisting) (i : Nat) :
    nativeRow B graphs c a i=nativeFormula B graphs c x entry i := by
  unfold nativeRow nativeFormula tileFormula
  rw [accumulate_entry B c _ compatible a]

def output (a : Allocated (BitVec 32) x entry preexisting) (final : State Payload) : Output (BitVec 32) :=
  ⟨x.rows,.fp32,x.device,1,a.base,a.commit (fun i => ((final.mem "C" i).read .fp32).bits)⟩

def execute (B : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (rt : Runtime (BitVec 32) x entry preexisting) : HostOutcome (BitVec 32) :=
  if x.columns≠x.vectorLength then .incompatibleDimensions else
    match rt.allocate with
    | none => .allocationFailure
    | some a => if rt.launchReady a then
        match runPlan (family B graphs) (rt.plan a) (liftAllocated a).state with
        | none => .sourceFailure
        | some final => .returned (output a final)
      else .setupFailure

/-- Actual full native graph for the selected source execution, after every
autotune trial. Only single-instruction FMA/add semantics are abstract; every
source load/cast/carry/reduction/store and physical frame is already connected. -/
theorem execute_returns (B : Scalars)
    (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (rt : Runtime (BitVec 32) x entry preexisting) (compatible : x.columns=x.vectorLength)
    (a : Allocated (BitVec 32) x entry preexisting) (allocated : rt.allocate=some a)
    (ready : rt.launchReady a=Bool.true) :
    ∃ out,execute B graphs rt=.returned out ∧ out.length=x.rows ∧ out.dtype=.fp32 ∧
      out.device=x.device ∧ out.stride=1 ∧ out.base=a.base ∧
      (∀ i,i<x.rows → out.memory (out.base+i)=nativeRow B graphs (rt.plan a).selected.config a i) ∧
      (∀ addr,(∀ i,i<x.rows → addr≠out.base+i) → out.memory addr=a.memory addr) ∧
      (∀ addr,preexisting addr → out.memory addr=entry addr) := by
  obtain ⟨final,ran,values,frame⟩ := plan_result (family B graphs) (rt.plan a) (liftAllocated a).state
  have exterior (i : Nat) (hi : x.rows ≤ i) :
      ((final.mem "C" i).read .fp32).bits=a.memory (a.base+i) := by
    rw [frame "C" i (Or.inr hi)]
    rfl
  have physicalFrame := a.commit_frame _ exterior
  refine ⟨output a final,?_,rfl,rfl,rfl,rfl,rfl,?_,physicalFrame,?_⟩
  · simp [execute,compatible,allocated,ready,ran]
  · intro i hi
    simp only [output]
    rw [a.commit_value,values i hi,source_row]
    rfl
  · intro addr hp
    simp only [output]
    rw [physicalFrame addr (by
      intro i hi same
      apply a.fresh i hi
      simpa only [←same] using hp)]
    exact a.protected_frame addr hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNativeHost
