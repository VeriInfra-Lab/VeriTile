import bench.tritonbench_g.chunk_linear_attn.WholeReal
import bench.tritonbench_g.chunk_linear_attn.AddressBounds
import bench.tritonbench_g.chunk_delta_fwd.Proofs.HostCall
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 10000000
abbrev Tensor := VeriTile.Bench.Repairs.ChunkDeltaFwd.Host.Tensor
abbrev denseStrides := VeriTile.Bench.Repairs.ChunkDeltaFwd.Host.denseStrides
abbrev span := VeriTile.Bench.Repairs.ChunkDeltaFwd.Host.span

inductive ScaleOrigin where
  | supplied (pythonValue : ℝ)
  | defaultPower (featureDimension : Nat)
structure ScaleBinding where
  origin : ScaleOrigin
  pythonBinary64 : BitVec 64
  argumentFP32 : BitVec 32
  scalarValue : ℝ

/-- Descriptor and routing metadata. FP32 scalar conversion is recorded by
ScaleBinding and remains a local scalar ABI obligation, not an output oracle. -/
structure Invocation where
  dims : Dimensions
  (query key value : Tensor)
  (initial incoming terminal : Option Tensor)
  (final normalize : Bool)
  scale : ScaleBinding
  selection : LibrarySelection (dims.parameters Bool.true)

def Invocation.mathCall (c : Invocation) : Real.Call :=
  ⟨c.dims,c.scale.scalarValue,c.initial.isSome,c.final,c.terminal.isSome,c.selection⟩
def Invocation.forwardInputs (c : Invocation) := [c.query,c.key,c.value]++c.initial.toList

/-- Source-visible bounds on every dimension used by the four actual grids. -/
structure LaunchGuard (d : Dimensions) : Prop where
  bh : d.B*d.H≤65535
  nt : d.NT≤65535
  nk : keyBlocks (d.parameters Bool.false)≤65535
  nvf : valueBlocks (d.parameters Bool.false)≤65535
  nvb : valueBlocks (d.parameters Bool.true)≤65535

def launchMax (d : Dimensions) : Nat :=
  max (d.B*d.H) (max d.NT (max (keyBlocks (d.parameters Bool.false))
    (max (valueBlocks (d.parameters Bool.false)) (valueBlocks (d.parameters Bool.true)))))

theorem source_launch_guard (d : Dimensions) : launchMax d≤65535 ↔ LaunchGuard d := by
  constructor
  · intro h
    simp only [launchMax,max_le_iff] at h
    exact ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2⟩
  · intro h
    simp only [launchMax,max_le_iff]
    exact ⟨h.bh,h.nt,h.nk,h.nvf,h.nvb⟩

structure Contract (c : Invocation) : Prop where
  valid : Valid c.dims
  machine : MachineGuard c.dims
  launch : LaunchGuard c.dims
  queryShape : c.query.shape=queryShape c.dims
  keyShape : c.key.shape=c.query.shape
  valueShape : c.value.shape=valueShape c.dims
  queryStrides : c.query.strides=inputStrides c.dims Bool.false
  keyStrides : c.key.strides=c.query.strides
  valueStrides : c.value.strides=inputStrides c.dims Bool.true
  initialShape : ∀ t∈c.initial, t.shape=stateShape c.dims ∧ t.contiguous=Bool.true
  incomingShape : ∀ t∈c.incoming, t.shape=Host.valueShape c.dims
  terminalShape : ∀ t∈c.terminal, t.shape=stateShape c.dims
  terminalRoute : c.terminal.isSome=Bool.true → c.final=Bool.true
  ordinary : ∀ t∈c.forwardInputs++c.incoming.toList++c.terminal.toList,
    t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧ t.device=c.query.device ∧ span t≤t.capacity
  scaleOrigin : match c.scale.origin with
    | .supplied _ => True
    | .defaultPower K => K=c.dims.K

def Invocation.parametersFor (c : Invocation) (stage : Stage) : Parameters :=
  c.dims.parameters (match stage with | .forwardState | .forwardOutput => Bool.false | _ => Bool.true)
def AxisSafe (g : PID) : Prop :=
  0<g.x ∧ g.x≤65535 ∧ 0<g.y ∧ g.y≤65535 ∧ 0<g.z ∧ g.z≤65535

/-- Every dimension of each actual Python grid is positive and at most 65535.
This is an axis bound, not a claim about generic launch-product arithmetic. -/
theorem launch_axes_safe (c : Invocation) (cc : Contract c) (stage : Stage) :
    AxisSafe (gridSize stage (c.parametersFor stage)) := by
  have hk : 0<keyBlocks (c.dims.parameters Bool.false) := by
    have hs := (covers c.dims cc.valid Bool.false).2.1
    have hm := cc.valid.k_min
    by_contra hn
    have hz : keyBlocks (c.dims.parameters Bool.false)=0 := by omega
    simp only [hz,Nat.zero_mul] at hs
    omega
  have hv : ∀ b, 0<valueBlocks (c.dims.parameters b) := by
    intro b
    have hs := (covers c.dims cc.valid b).2.2
    have hm := cc.valid.v_min
    by_contra hn
    have hz : valueBlocks (c.dims.parameters b)=0 := by omega
    simp only [hz,Nat.zero_mul] at hs
    omega
  have ht : 0<c.dims.NT := by
    have hs := (covers c.dims cc.valid Bool.false).1
    have hm := cc.valid.t_pos
    change c.dims.T≤c.dims.NT*64 at hs
    omega
  have hbh := Nat.mul_pos cc.valid.b_pos cc.valid.h_pos
  cases stage with
  | forwardState => exact ⟨hk,cc.launch.nk,hv Bool.false,cc.launch.nvf,hbh,cc.launch.bh⟩
  | forwardOutput => exact ⟨hv Bool.false,cc.launch.nvf,ht,cc.launch.nt,hbh,cc.launch.bh⟩
  | backwardState => exact ⟨hk,cc.launch.nk,hv Bool.true,cc.launch.nvb,hbh,cc.launch.bh⟩
  | backwardGradient => exact ⟨hk,cc.launch.nk,ht,cc.launch.nt,hbh,cc.launch.bh⟩

theorem machine_launch_axes_safe {α : Type} [Inhabited α] (c : Invocation) (cc : Contract c)
    (stage : Stage) (pid : PID) (m : Memory α) :
    let s := launchState stage (c.parametersFor stage) pid m
    0<s.numPids 0 ∧ s.numPids 0≤65535 ∧ 0<s.numPids 1 ∧ s.numPids 1≤65535 ∧
      0<s.numPids 2 ∧ s.numPids 2≤65535 :=
  launch_axes_safe c cc stage

structure Allocations where
  (history output reverseHistory queryGradient keyGradient valuePartials valueGradient incomingDense : Tensor)
  (finalState initialGradient terminalDense : Option Tensor)

def Allocations.fresh (a : Allocations) :=
  [a.history,a.output,a.reverseHistory,a.queryGradient,a.keyGradient,a.valuePartials,a.valueGradient]++
    a.finalState.toList++a.initialGradient.toList
structure AllocationContract (c : Invocation) (a : Allocations) : Prop where
  historyShape : a.history.shape=Host.historyShape c.dims
  outputShape : a.output.shape=Host.valueShape c.dims
  reverseShape : a.reverseHistory.shape=Host.historyShape c.dims
  queryShape : a.queryGradient.shape=Host.queryShape c.dims
  keyShape : a.keyGradient.shape=Host.queryShape c.dims
  partialShape : a.valuePartials.shape=Host.partialShape c.dims
  valueShape : a.valueGradient.shape=Host.valueShape c.dims
  finalFlag : a.finalState.isSome=c.final
  initialFlag : a.initialGradient.isSome=c.initial.isSome
  terminalFlag : a.terminalDense.isSome=c.terminal.isSome
  stateShapes : ∀ t∈a.finalState.toList++a.initialGradient.toList++a.terminalDense.toList, t.shape=stateShape c.dims
  incomingShape : a.incomingDense.shape=Host.valueShape c.dims
  descriptors : ∀ t∈a.fresh++[a.incomingDense]++a.terminalDense.toList,
    t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧ t.device=c.query.device ∧
      t.strides=denseStrides t.shape ∧ t.contiguous=Bool.true ∧ t.shape.prod≤t.capacity
  freshIdentity : ∀ t∈a.fresh, ∀ u∈c.forwardInputs++c.incoming.toList++c.terminal.toList, t.identity≠u.identity
  pairIdentity : ∀ t∈a.fresh, ∀ u∈a.fresh, t≠u → t.identity≠u.identity

/-- Ordinary view/copy semantics at individual indices. Incoming cotangents
may have zero strides; canonical forward tensors are never copied. -/
def offset4 (t : Tensor) (b h time feature : Nat) : Nat :=
  b*t.strides[0]!+h*t.strides[1]!+time*t.strides[2]!+feature*t.strides[3]!
structure CotangentCopies {α : Type} (M : Algebra α) (c : Invocation) (a : Allocations)
    (physical : Nat → Cell α) : Prop where
  incoming : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V),
    physical (a.incomingDense.base+((b.val*c.dims.H+h.val)*c.dims.T+t.val)*c.dims.V+j.val)=
      match c.incoming with
      | none => .mk .real (zero M)
      | some src => physical (src.base+offset4 src b.val h.val t.val j.val)
  terminal : ∀ src∈c.terminal, ∀ dst∈a.terminalDense,
    ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V),
      physical (dst.base+((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val)=
        physical (src.base+offset4 src b.val h.val e.val j.val)

def descriptor (c : Invocation) (a : Allocations) (r : RegionName) : Option Tensor :=
  if r="query" then some c.query else if r="key" then some c.key else if r="value" then some c.value
  else if r="initial" then c.initial else if r="incoming" then some a.incomingDense
  else if r="terminalGradient" then a.terminalDense else if r="history" then some a.history
  else if r="output" then some a.output else if r="historyGradient" then some a.reverseHistory
  else if r="queryGradient" then some a.queryGradient else if r="keyGradient" then some a.keyGradient
  else if r="valuePartials" then some a.valuePartials else if r="valueGradient" then some a.valueGradient
  else if r="final" then a.finalState else if r="initialGradient" then a.initialGradient else none
def logical {α : Type} (c : Invocation) (a : Allocations) (physical : Nat → Cell α) (fallback : Memory α) : Memory α :=
  fun r o => match descriptor c a r with | some t => physical (t.base+o) | none => fallback r o

structure Context where
  (query key value history : Tensor)
  scale : ScaleBinding
  (hasInitial materializeGrads : Bool)
def saveContext (c : Invocation) (a : Allocations) : Context :=
  ⟨c.query,c.key,c.value,a.history,c.scale,c.initial.isSome,Bool.false⟩
def returnedForward (a : Allocations) := (a.output,a.finalState)
def returnedBackward (a : Allocations) : List (Option Tensor) :=
  [some a.queryGradient,some a.keyGradient,some a.valueGradient,none,a.initialGradient,none]
structure Result (α : Type) where
  memory : Memory α
  forward : Tensor × Option Tensor
  backward : List (Option Tensor)
  context : Context

def execute {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : Invocation) (a : Allocations) (m : Memory α) : Option (Result α) := do
  let f := c.dims.parameters Bool.false
  let b := c.dims.parameters Bool.true
  let kernels ← executeFour M D f b c.scale.scalarValue c.initial.isSome c.final c.terminal.isSome
    (grid .forwardState f) (grid .forwardOutput f) (grid .backwardState b) (grid .backwardGradient b) m
  return ⟨hostSumMemory M b c.selection kernels kernels,returnedForward a,returnedBackward a,saveContext c a⟩

theorem execute_complete {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : Invocation) (a : Allocations) (m : Memory α) :
    execute M D c a m=some ⟨completeMemory M D c.mathCall.forward c.mathCall.backward c.scale.scalarValue
      c.initial.isSome c.final c.terminal.isSome c.selection m,returnedForward a,returnedBackward a,saveContext c a⟩ := by
  simp only [execute,four_execution,Option.bind_some,bind,completeMemory,Invocation.mathCall,Real.Call.forward,Real.Call.backward]
  rfl

theorem returned_slots (a : Allocations) :
    (returnedBackward a).length=6 ∧ (returnedBackward a)[4]?=some a.initialGradient ∧
    (returnedBackward a)[3]?=some none ∧ (returnedBackward a)[5]?=some none := by
  simp [returnedBackward]

theorem normalize_noop_execute {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : Invocation) (a : Allocations) (m : Memory α) :
    execute M D {c with normalize:=Bool.true} a m=execute M D {c with normalize:=Bool.false} a m := rfl

theorem saved_context (c : Invocation) (a : Allocations) :
    (saveContext c a).query=c.query ∧ (saveContext c a).key=c.key ∧
    (saveContext c a).value=c.value ∧ (saveContext c a).history=a.history ∧
    (saveContext c a).scale=c.scale ∧ (saveContext c a).hasInitial=c.initial.isSome ∧
    (saveContext c a).materializeGrads=Bool.false := by simp [saveContext]

theorem real_output (c : Invocation) (contract : Contract c) (a : Allocations) (m : Memory ℝ)
    (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V) :
    ∃ result, execute Real.algebra Real.dotPrimitive c a m=some result ∧
      result.forward=returnedForward a ∧ result.backward=returnedBackward a ∧
      result.memory "output" ((head*c.dims.T+t.val)*c.dims.V+j.val)=
        .mk .real (Real.output c.scale.scalarValue (Real.inputs c.mathCall.forward head m c.initial.isSome) t j) := by
  refine ⟨⟨c.mathCall.result m,returnedForward a,returnedBackward a,saveContext c a⟩,?_,rfl,rfl,?_⟩
  · exact execute_complete Real.algebra Real.dotPrimitive c a m
  · exact c.mathCall.output_read contract.valid m head hh t j

#axiomsClean execute_complete
#axiomsClean returned_slots
#axiomsClean normalize_noop_execute
#axiomsClean saved_context
#axiomsClean real_output
#axiomsClean source_launch_guard
#axiomsClean launch_axes_safe
#axiomsClean machine_launch_axes_safe
#specNonCircular LaunchGuard avoiding [execute,gridMemory,Real.Call.result]
#specNonCircular Contract avoiding [execute,gridMemory,Real.Call.result]
#specNonCircular AllocationContract avoiding [execute,gridMemory,Real.Call.result]
#specNonCircular CotangentCopies avoiding [execute,gridMemory,Real.Call.result,Real.gradients]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
