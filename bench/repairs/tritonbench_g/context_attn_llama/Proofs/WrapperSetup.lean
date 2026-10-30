import bench.repairs.tritonbench_g.context_attn_llama.Proofs.WrapperPrelude

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 3000000

abbrev Key := RegionName × Nat
abbrev Memory := Key → Cell (BitVec 32)

/-- Coordinates of a tensor before the first normalization operation. The
address function includes its actual storage offset/strides; a resolved
buffer may use a different offset encoding for the same logical coordinate. -/
structure View where
  address : Nat → Key
  negative : Bool

def View.word (v : View) (before : Memory) (n : Nat) : BitVec 32 :=
  FPViews.sign v.negative ((before (v.address n)).read .real)
def View.integer (v : View) (before : Memory) (n : Nat) : Int :=
  if v.negative then -((before (v.address n)).read .int) else (before (v.address n)).read .int

structure Initial (c : Host.Call) where
  before : Memory
  query : View
  key : View
  value : View
  start : View
  sequence : View
  prompt : View
  request : View
  pages : View
  output : View
  start_value : ∀ b, b < c.batches → start.integer before b = (c.start b : Int)
  sequence_value : ∀ b, b < c.batches → sequence.integer before b = ((c.length b+c.prompt b : Nat) : Int)
  prompt_value : ∀ b, b < c.batches → prompt.integer before b = (c.prompt b : Int)
  request_value : ∀ b, b < c.batches → request.integer before b = (c.request b : Int)
  pages_value : ∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b →
    pages.integer before (c.layout.reqRow*c.request b+c.layout.reqToken*j) = (c.page b j : Int)

/-- Both descriptions name the original supplied output object, before clone
changes the local `o` binding. The address relation follows tensor coordinates
through the clone's possibly changed strides. -/
structure CallerOutput (c : Host.Call) (input : Initial c)
    (o : FPViews.Output (FPHost.Coordinate c)) : Prop where
  before : o.initial = input.before
  negative : o.negative = input.output.negative
  coordinates : ∀ i, input.output.address (c.output i.1 i.2.1 i.2.2.1 i.2.2.2) = o.address i

structure Runtime where
  kernel : State NativeGraph.Expr
  caller : Memory
  starts : List Int
  sequences : List Int
  prompts : List Int
  descriptor : Stage → Option Nat

/-- Values returned by the actual ordered stride reads on resolved Q/K/V,
the private output clone, and the resolved page tensor. -/
def expectedStride (c : Host.Call) : Stage → Option Nat
  | .qStride0 => some c.layout.qRow | .qStride1 => some c.layout.qHead | .qStride2 => some c.layout.qDim
  | .kStride0 => some c.layout.kRow | .kStride1 => some c.layout.kHead | .kStride2 => some c.layout.kDim
  | .vStride0 => some c.layout.vRow | .vStride1 => some c.layout.vHead | .vStride2 => some c.layout.vDim
  | .outStride0 => some c.layout.outRow | .outStride1 => some c.layout.outHead | .outStride2 => some c.layout.outDim
  | .pagesStride0 => some c.layout.reqRow | .pagesStride1 => some c.layout.reqToken
  | _ => none

def Descriptors (c : Host.Call) (s : Runtime) : Prop :=
  ∀ stage value, expectedStride c stage = some value → s.descriptor stage = some value

def words (c : Host.Call) (v : View) (initial : Memory) (s : Runtime) (rg : RegionName) : Prop :=
  ∀ n, (s.kernel.mem rg n).read .real = .leaf (v.word initial n)
def integers (v : View) (initial : Memory) (s : Runtime) (rg : RegionName) : Prop :=
  ∀ n, (s.kernel.mem rg n).read .int = v.integer initial n

/-- Each successful local API operation establishes only its own coordinate
contract or snapshot. None of these predicates mentions kernel execution,
expected output, a global prepared state, or a floating arithmetic law. -/
def Bound (c : Host.Call) (input : Initial c) : Stage → Runtime → Prop
  | .resolveQ,s => words c input.query input.before s c.regions.query
  | .resolveK,s => words c input.key input.before s c.regions.key
  | .resolveV,s => words c input.value input.before s c.regions.value
  | .resolveStart,s | .contiguousStart,s => integers input.start input.before s c.regions.start
  | .resolveSequence,s | .contiguousSequence,s => integers input.sequence input.before s c.regions.sequence
  | .resolvePrompt,s | .contiguousPrompt,s => integers input.prompt input.before s c.regions.prompt
  | .resolveRequest,s | .contiguousRequest,s => integers input.request input.before s c.regions.request
  | .resolvePages,s => integers input.pages input.before s c.regions.pages
  | .resolveOut,s | .cloneOut,s => words c input.output input.before s c.regions.out
  | .tolistStart,s => s.starts = List.ofFn (fun b : Fin c.batches => input.start.integer input.before b)
  | .tolistSequence,s => s.sequences = List.ofFn (fun b : Fin c.batches => input.sequence.integer input.before b)
  | .tolistPrompt,s => s.prompts = List.ofFn (fun b : Fin c.batches => input.prompt.integer input.before b)
  | stage,s => ∀ value, expectedStride c stage = some value → s.descriptor stage = some value

structure LocalAPI (c : Host.Call) (input : Initial c) where
  step : Stage → Runtime → StepResult Runtime
  establish : ∀ stage s t, step stage s = .ok t → Bound c input stage t
  preserve : ∀ stage s other, other ≠ stage → Bound c input other s →
    Bound c input other (step stage s).state
  frame : ∀ stage s cell, (step stage s).state.caller cell = s.caller cell

def LocalAPI.api (api : LocalAPI c input) : API Runtime Key (Cell (BitVec 32)) where
  step := api.step
  observe := Runtime.caller
  caller := fun _ => True
  binds := Bound c input
  establish := api.establish
  preserve := api.preserve
  frame := fun stage s cell _ => api.frame stage s cell

theorem prepared_metadata (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (s t : Runtime) (d : Dimensions)
    (h : runPrefix api.api request s = .ready d t) : Native.FullMetadata c t.kernel := by
  have hb := (prefix_binds api.api request s t d h).2
  have hs : integers input.start input.before t c.regions.start := hb .contiguousStart (by simp [normalization,setup])
  have hl : integers input.sequence input.before t c.regions.sequence := hb .contiguousSequence (by simp [normalization,setup])
  have hp : integers input.prompt input.before t c.regions.prompt := hb .contiguousPrompt (by simp [normalization,setup])
  have hr : integers input.request input.before t c.regions.request := hb .contiguousRequest (by simp [normalization,setup])
  have hpg : integers input.pages input.before t c.regions.pages := hb .resolvePages (by simp [normalization,setup])
  refine ⟨⟨?_,?_,?_⟩,?_,?_⟩
  · intro b hbound; exact (hs b).trans (input.start_value b hbound)
  · intro b hbound; exact (hl b).trans (input.sequence_value b hbound)
  · intro b hbound; exact (hp b).trans (input.prompt_value b hbound)
  · intro b hbound; exact (hr b).trans (input.request_value b hbound)
  · intro b hbound j hj; exact (hpg _).trans (input.pages_value b hbound j hj)

theorem prepared_words (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (s t : Runtime) (d : Dimensions)
    (h : runPrefix api.api request s = .ready d t) :
    words c input.query input.before t c.regions.query ∧
    words c input.key input.before t c.regions.key ∧
    words c input.value input.before t c.regions.value ∧
    words c input.output input.before t c.regions.out := by
  have hb := (prefix_binds api.api request s t d h).2
  exact ⟨hb .resolveQ (by simp [normalization,setup]),hb .resolveK (by simp [normalization,setup]),
    hb .resolveV (by simp [normalization,setup]),hb .cloneOut (by simp [normalization,setup])⟩

theorem prepared_descriptors (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (s t : Runtime) (d : Dimensions)
    (h : runPrefix api.api request s = .ready d t) : Descriptors c t := by
  have hb := (prefix_binds api.api request s t d h).2
  intro stage value hv
  have bound : Bound c input stage t := hb stage (by cases stage <;> simp [normalization,setup])
  cases stage <;> simp [expectedStride] at hv
  all_goals subst value
  all_goals exact bound _ rfl

theorem initialized_clone (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (s t : Runtime) (d : Dimensions)
    (h : runPrefix api.api request s = .ready d t)
    (o : FPViews.Output (FPHost.Coordinate c)) (output : CallerOutput c input o) :
    ∀ i, (t.kernel.mem c.regions.out (c.output i.1 i.2.1 i.2.2.1 i.2.2.2)).read .real =
      .leaf (o.logical input.before i) := by
  intro i
  have hword := (prepared_words c input api request s t d h).2.2.2
  rw [hword]
  simp only [View.word,output.coordinates,FPViews.Output.logical,output.negative]

/-- The snapshots are computed before launch/caller stores, in Python's exact
start, sequence, prompt order. Their lengths agree by the ordinary metadata
geometry, so zip visits exactly the source batch order. -/
theorem prepared_snapshots (c : Host.Call) (input : Initial c) (api : LocalAPI c input)
    (request : Request) (s t : Runtime) (d : Dimensions)
    (h : runPrefix api.api request s = .ready d t) :
    t.starts = List.ofFn (fun b : Fin c.batches => (c.start b : Int)) ∧
    t.sequences = List.ofFn (fun b : Fin c.batches => ((c.length b+c.prompt b : Nat) : Int)) ∧
    t.prompts = List.ofFn (fun b : Fin c.batches => (c.prompt b : Int)) := by
  have hb := (prefix_binds api.api request s t d h).2
  have hs : Bound c input .tolistStart t := hb .tolistStart (by simp [normalization,setup])
  have hl : Bound c input .tolistSequence t := hb .tolistSequence (by simp [normalization,setup])
  have hp : Bound c input .tolistPrompt t := hb .tolistPrompt (by simp [normalization,setup])
  simp only [Bound] at hs hl hp
  refine ⟨hs.trans ?_,hl.trans ?_,hp.trans ?_⟩ <;> congr 1 <;> funext b
  · exact input.start_value b b.isLt
  · exact input.sequence_value b b.isLt
  · exact input.prompt_value b b.isLt

/-- Python's max(seq-prompt,0), followed by the covered-row minimum. This
equality is pure integer arithmetic, not an API or numerical assumption. -/
theorem captured_rows (c : Host.Call) (b : Nat) :
    min (max (((c.length b+c.prompt b : Nat) : Int)-(c.prompt b : Int)) 0).toNat
      (c.blocks*c.layout.BM) = FPHost.copyRows c b := by
  simp [FPHost.copyRows,Nat.cast_add]

#print axioms prepared_metadata
#print axioms prepared_words
#print axioms prepared_descriptors
#print axioms prepared_snapshots
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
