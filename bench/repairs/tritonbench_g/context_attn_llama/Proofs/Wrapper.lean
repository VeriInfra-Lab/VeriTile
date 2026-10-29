import bench.repairs.tritonbench_g.context_attn_llama.Proofs.WrapperCopy

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 5000000

/-- Source descriptor agreement, not successful execution. Invalid shape
requests still follow the actual assertion/error path. A valid descriptor uses
the passed max_input_len and Tesla branch, and the actual grouping quotient. -/
structure Geometry (request : Request) (c : Host.Call) : Prop where
  validated : ∀ d, validate request = .ok d →
    d.width = c.layout.D ∧ d.batches = c.batches ∧ d.heads = c.layout.heads ∧ d.kvGroup = c.layout.kvGroup
  tile : c.layout.BM = if request.tesla then 64 else 128
  blocks : c.blocks = request.maxInputLen / c.layout.BM +
    if request.maxInputLen % c.layout.BM = 0 then 0 else 1

inductive Outcome (c : Host.Call) where
  | invalid (error : ValidationError) (state : Runtime)
  | setupFailed (stage : Stage) (error : String) (state : Runtime)
  | kernelFailure (prepared : Runtime)
  | copyFailed (prepared : Runtime) (graph : State NativeGraph.Expr)
      (batch : Fin c.batches) (site : CopySite) (error : String) (memory : Memory)
  | returned (prepared : Runtime) (graph : State NativeGraph.Expr) (memory : Memory)

def Outcome.memory {c : Host.Call} : Outcome c → Memory
  | .invalid _ s | .setupFailed _ _ s | .kernelFailure s => s.caller
  | .copyFailed _ _ _ _ _ mem | .returned _ _ mem => mem

def execute (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial : Runtime) (ops : NativeGraph.Operations) (plan : NativeGraph.Plan)
    (owner : Coordinate c → NativeGraph.Owner) (o : FPViews.Output (Coordinate c))
    (copy : ∀ t : State NativeGraph.Expr, CopyAPI c o (NativeGraph.scratch ops plan c owner t)) : Outcome c :=
  match runPrefix api.api request initial with
  | .invalid error state => .invalid error state
  | .failed stage error state => .setupFailed stage error state
  | .ready _ prepared =>
    match FPSuccess.runGrid c (NativeGraph.backend plan) (FPSuccess.allPrograms c) prepared.kernel with
    | none => .kernelFailure prepared
    | some t => match copies (copy t) (List.finRange c.batches) prepared.caller with
      | .failed b site error memory => .copyFailed prepared t b site error memory
      | .ok memory => .returned prepared t memory

/-- The same caller cell frame holds at every outcome, including partial
copy failures. Cells inside an attempted output slice may have changed on
failure; no fictitious rollback or blanket pre-call state is substituted. -/
theorem whole_frame (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial : Runtime) (ops : NativeGraph.Operations) (plan : NativeGraph.Plan)
    (owner : Coordinate c → NativeGraph.Owner) (o : FPViews.Output (Coordinate c))
    (copy : ∀ t : State NativeGraph.Expr, CopyAPI c o (NativeGraph.scratch ops plan c owner t))
    (cell : Key) (hmiss : ¬∃ i, active c i ∧ o.address i = cell) :
    (execute c input api request initial ops plan owner o copy).memory cell = initial.caller cell := by
  have hp := prefix_frame api.api request initial cell trivial
  change (runPrefix api.api request initial).state.caller cell = initial.caller cell at hp
  cases hx : runPrefix api.api request initial with
  | invalid error state => simpa only [execute,hx,Outcome.memory,PrefixResult.state] using hp
  | failed stage error state => simpa only [execute,hx,Outcome.memory,PrefixResult.state] using hp
  | ready d prepared =>
    have hf : prepared.caller cell = initial.caller cell := by simpa only [hx,PrefixResult.state] using hp
    cases hg : FPSuccess.runGrid c (NativeGraph.backend plan) (FPSuccess.allPrograms c) prepared.kernel with
    | none => simpa [execute,hx,hg,Outcome.memory] using hf
    | some t =>
      have hc := copies_frame (copy t) (List.finRange c.batches) prepared.caller cell hmiss
      have joined := hc.trans hf
      cases hy : copies (copy t) (List.finRange c.batches) prepared.caller <;>
        simpa only [execute,hx,hg,hy,Outcome.memory,CopyResult.memory] using joined

def ReachedSetup (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial prepared : Runtime) : Prop :=
  ∃ d, runPrefix api.api request initial = .ready d prepared ∧
    d.width = c.layout.D ∧ d.batches = c.batches ∧ d.heads = c.layout.heads ∧ d.kvGroup = c.layout.kvGroup

def Captured (c : Host.Call) (prepared : Runtime) : Prop :=
  prepared.starts = List.ofFn (fun b : Fin c.batches => (c.start b : Int)) ∧
  prepared.sequences = List.ofFn (fun b : Fin c.batches => ((c.length b+c.prompt b : Nat) : Int)) ∧
  prepared.prompts = List.ofFn (fun b : Fin c.batches => (c.prompt b : Int))

def GraphResult (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial prepared : Runtime) (plan : NativeGraph.Plan)
    (t : State NativeGraph.Expr) (A : FlatAlloc) (logical : BlockState)
    (w : Native.Widths) (B : Native.Bytes A) : Prop :=
  ReachedSetup c input api request initial prepared ∧
  Native.FullMetadata c prepared.kernel ∧
  words c input.query input.before prepared c.regions.query ∧
  words c input.key input.before prepared c.regions.key ∧
  words c input.value input.before prepared c.regions.value ∧
  words c input.output input.before prepared c.regions.out ∧
  Captured c prepared ∧
  Descriptors c prepared ∧
  Native.GridBridge c A logical w B (NativeGraph.backend plan) prepared.kernel ∧
  FPSuccess.runGrid c (NativeGraph.backend plan) (FPSuccess.allPrograms c) prepared.kernel = some t ∧
  (∀ rg n, ¬ FPSuccess.writes c (FPSuccess.allPrograms c) rg n → t.mem rg n = prepared.kernel.mem rg n)

def Contract (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (initial : Runtime) (ops : NativeGraph.Operations) (plan : NativeGraph.Plan)
    (owner : Coordinate c → NativeGraph.Owner) (o : FPViews.Output (Coordinate c))
    (copy : ∀ t : State NativeGraph.Expr, CopyAPI c o (NativeGraph.scratch ops plan c owner t))
    (A : FlatAlloc) (logical : BlockState) (w : Native.Widths) (B : Native.Bytes A) : Prop :=
    (∀ cell, (¬∃ i, active c i ∧ o.address i = cell) →
      (execute c input api request initial ops plan owner o copy).memory cell = input.before cell) ∧
    match execute c input api request initial ops plan owner o copy with
    | .invalid error state =>
        steps api.api normalization initial = .ok state ∧ validate request = .error error ∧ state.caller = input.before
    | .setupFailed stage error state =>
        (FailedAt api.api normalization initial stage error state ∨
          ∃ normalized d, steps api.api normalization initial = .ok normalized ∧ validate request = .ok d ∧
            FailedAt api.api setup normalized stage error state) ∧ state.caller = input.before
    | .kernelFailure _ => False
    | .copyFailed prepared t b site error mem =>
        GraphResult c input api request initial prepared plan t A logical w B ∧
        CopyFailedAt (copy t) (List.finRange c.batches) prepared.caller b site error mem
    | .returned prepared t mem =>
        GraphResult c input api request initial prepared plan t A logical w B ∧
        (∀ i, active c i → o.logical mem i = NativeGraph.scratch ops plan c owner t i) ∧
        ((none : Option Unit),o.owner,mem).1 = none ∧
        ((none : Option Unit),o.owner,mem).2.1 = o.owner

/-- Full source-wrapper protocol from the pre-normalization caller state.
The original logical source invocation is the FP reference. Setup callbacks
are local ordinary API contracts; successful source execution is derived.
The return branch reads the complete native graph through actual copyback. -/
theorem whole_wrapper (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (geometry : Geometry request c) (initial : Runtime)
    (initial_memory : initial.caller = input.before)
    (ops : NativeGraph.Operations) (plan : NativeGraph.Plan)
    (owner : Coordinate c → NativeGraph.Owner) (o : FPViews.Output (Coordinate c))
    (copy : ∀ t : State NativeGraph.Expr, CopyAPI c o (NativeGraph.scratch ops plan c owner t))
    (unique : ∀ i j, active c i → active c j → o.address i = o.address j → i = j)
    (A : FlatAlloc) (logical : BlockState) (valid : Host.Valid c A logical)
    (w : Native.Widths) (B : Native.Bytes A) (abi : Native.ABI c A w B) :
    Contract c input api request initial ops plan owner o copy A logical w B := by
  unfold Contract
  refine ⟨?_,?_⟩
  · intro cell hmiss
    exact (whole_frame c input api request initial ops plan owner o copy cell hmiss).trans (congrFun initial_memory cell)
  · have protocol := prefix_protocol api.api request initial
    have frame : (runPrefix api.api request initial).state.caller = input.before := by
      funext cell
      exact (prefix_frame api.api request initial cell trivial).trans (congrFun initial_memory cell)
    cases hx : runPrefix api.api request initial with
    | invalid error state =>
      simp only [hx,PrefixResult.state] at protocol frame
      simpa only [execute,hx] using ⟨protocol.1,protocol.2,frame⟩
    | failed stage error state =>
      simpa [execute,hx] using ⟨(by simpa [hx] using protocol), (by simpa [hx] using frame)⟩
    | ready d prepared =>
      have hm := prepared_metadata c input api request initial prepared d hx
      have hw := prepared_words c input api request initial prepared d hx
      have dimensions := geometry.validated d (prefix_binds api.api request initial prepared d hx).1
      have snapshots := prepared_snapshots c input api request initial prepared d hx
      have descriptors := prepared_descriptors c input api request initial prepared d hx
      obtain ⟨bridge,t,ht,hf⟩ := NativeGraph.successful_grid plan c A logical valid w B abi prepared.kernel hm
      have result : GraphResult c input api request initial prepared plan t A logical w B :=
        ⟨⟨d,hx,dimensions⟩,hm,hw.1,hw.2.1,hw.2.2.1,hw.2.2.2,snapshots,descriptors,bridge,ht,hf⟩
      cases hc : copies (copy t) (List.finRange c.batches) prepared.caller with
      | failed b site error mem =>
        simpa [execute,hx,ht,hc] using ⟨result,copies_failure (copy t) _ _ _ _ _ _ hc⟩
      | ok mem =>
        have hread := (complete_copy (copy t) prepared.caller mem unique hc).1
        simp only [execute,hx,ht,hc]
        exact ⟨result,hread,trivial,trivial⟩

#print axioms whole_wrapper
#print axioms whole_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
