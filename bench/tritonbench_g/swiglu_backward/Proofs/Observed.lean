import bench.tritonbench_g.swiglu_backward.Proofs.WholeIndex

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Observed
open VeriTile.Triton FP.Structural
open Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Operation nodes emitted by the original typed source interpreter. Input
memory contains only atoms; dtype and precision tags remain on every node. -/
inductive Term (α : Type) where
  | input (value : α)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Term α)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Term α)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
      (left right : Term α)
  | unused

instance : Inhabited (Term α) := ⟨.unused⟩

/-- Original complete source semantics produces the operation graph. This
backend supplies constructors, never computed output or a native oracle. -/
def graphAlgebra : Algebra (Term α) where
  literal := Term.literal
  negInf := .unused
  binary := Term.binary
  unary := Term.unary
  cast := Term.cast
  fromNat := fun _ _ => .unused
  fromInt := fun _ _ => .unused
  fp32Bits := fun _ => .unused
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => .unused
  reduceSum := fun _ {_} _ _ _ => fun _ => .unused

/-- Scalar primitive ports for the observed CUDA graph. `fma` is genuinely
ternary: no split multiply/add equality is assumed. `sigmoid` binds the entire
local native sigmoid lowering, including its exp/reciprocal approximations. -/
structure NativeModel (α : Type) where
  literal : ℝ → α
  cast : α → α
  sigmoid : α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  unused : α

/-- Task-local compiler-observed interpretation. The exact FP32 affine node
1+x*(1-sigmoid(x)) contracts to fma(1-sigmoid(x),x,1); x*sigmoid(x) has the
observed operand order sigmoid(x),x. These are operational rules, not numeric
identities. Other unmatched floating nodes have no claimed native meaning. -/
def interpret (M : NativeModel α) : Term α → α
  | .input value => value
  | .literal (some .fp32) .real value => M.literal value
  | .cast (some .fp32) .real .real value => M.cast (interpret M value)
  | .unary (some .fp32) .sigmoid value => M.sigmoid (interpret M value)
  | .binary (some .fp32) .real .add left
      (.binary (some .fp32) .real .mul a b) => M.fma (interpret M b) (interpret M a) (interpret M left)
  | .binary (some .fp32) .real .mul left (.unary (some .fp32) .sigmoid right) =>
      M.mul (M.sigmoid (interpret M right)) (interpret M left)
  | .binary (some .fp32) .real .mul left right => M.mul (interpret M left) (interpret M right)
  | .binary (some .fp32) .real .sub left right => M.sub (interpret M left) (interpret M right)
  | _ => M.unused

/-- Independent native scalar recurrence, read directly from the pinned
compiler graph. It is deliberately separate from the source-node evaluator. -/
def reference (M : NativeModel α) (ch : Channel) (x y d : α) : α :=
  let x := M.cast x
  let y := M.cast y
  let d := M.cast d
  let s := M.sigmoid x
  match ch with
  | .dx => M.mul (M.mul (M.mul s (M.fma (M.sub (M.literal 1) s) x (M.literal 1))) y) d
  | .dy => M.mul (M.mul s x) d
  | .out => M.mul (M.mul s x) y

/-- Concrete lowering of every output graph produced by the original source;
no relation between native FMA and separately rounded operations is needed. -/
theorem source_graph_lowering (M : NativeModel α) (ch : Channel) (x y d : α) :
    interpret M (Whole.scalar graphAlgebra ch (.input x) (.input y) (.input d)) =
      reference M ch x y d := by cases ch <;> rfl

/-- Exact-real projection of the observed graph has the independent SwiGLU
VJP and optional forward value. This does not identify FP32 with Real. -/
def realModel : NativeModel ℝ where
  literal := id
  cast := id
  sigmoid := Real.sigmoid
  sub := fun x y => x-y
  mul := fun x y => x*y
  fma := fun x y z => x*y+z
  unused := 0

theorem reference_real (ch : Channel) (x y d : ℝ) :
    reference realModel ch x y d = Whole.formula ch x y d := by
  cases ch <;> simp only [reference,realModel,Whole.formula,Math.dx,Math.dy,Math.forward,
    Math.sigmoid_eq,id_eq] <;> ring

def entryGraph (entry : PhysicalMemory α) : PhysicalMemory (Term α) :=
  fun allocation a => .input (entry allocation a)

def projectMemory (M : NativeModel α) (memory : PhysicalMemory (Term α)) : PhysicalMemory α :=
  fun allocation a => interpret M (memory allocation a)

def projectOutcome (M : NativeModel α) : Whole.Outcome (Term α) → Whole.Outcome α
  | .preparationFailure reason => .preparationFailure reason
  | .assertionFailure => .assertionFailure
  | .runtimeFailure reason => .runtimeFailure reason
  | .kernelFailure => .kernelFailure
  | .returned gradient output memory => .returned gradient output (projectMemory M memory)

def executeHost (M : NativeModel α)
    (rt : Whole.Runtime (Term α) call (entryGraph entry) preexisting) : Whole.Outcome α :=
  projectOutcome M (Whole.executeHost graphAlgebra rt)

def ReturnedSpec (M : NativeModel α)
    (p : Prepared (Term α) call (entryGraph entry) preexisting) (final : State (Term α)) : Prop :=
  p.returnGradient.shape = call.input.shape ∧
  (call.recompute = Bool.false → p.returnOutput = none) ∧
  (call.recompute = Bool.true → ∃ out, p.returnOutput = some out ∧ out.shape = p.batchShape++[p.ncols]) ∧
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    projectMemory M (p.commit final) (p.location ch r q).1 (p.location ch r q).2 = reference M ch
      (call.input.read entry (r*(2*p.ncols)+q))
      (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
      (call.upstream.read entry (r*p.ncols+q))) ∧
  (∀ region a, WholeOutside p.layout call.recompute region a →
    final.mem region a = p.state.mem region a) ∧
  (∀ allocation a, PhysicalOutside p allocation a →
    projectMemory M (p.commit final) allocation a = projectMemory M p.memory allocation a) ∧
  (∀ allocation a, preexisting allocation → PhysicalOutside p allocation a →
    projectMemory M (p.commit final) allocation a = entry allocation a) ∧
  (∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.xy.matrix.read (projectMemory M (p.commit final)) r q =
      p.xy.matrix.read (projectMemory M p.memory) r q) ∧
  (∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.dout.matrix.read (projectMemory M (p.commit final)) r q =
      p.dout.matrix.read (projectMemory M p.memory) r q)

def HostSpec (M : NativeModel α) (rt : Whole.Runtime (Term α) call (entryGraph entry) preexisting)
    (outcome : Whole.Outcome α) : Prop :=
  (∃ reason, rt.prepare = .error reason ∧ outcome = .preparationFailure reason) ∨
  (∃ p, rt.prepare = .ok p ∧ checks p = Bool.false ∧ outcome = .assertionFailure) ∨
  (∃ p reason, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .error reason ∧
    outcome = .runtimeFailure reason) ∨
  (∃ p plan final, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .ok plan ∧
    outcome = .returned p.returnGradient p.returnOutput (projectMemory M (p.commit final)) ∧
    ReturnedSpec M p final)

theorem returned_projection (M : NativeModel α)
    (p : Prepared (Term α) call (entryGraph entry) preexisting) (final : State (Term α))
    (h : Whole.ReturnedSpec (Whole.scalar graphAlgebra) p final) : ReturnedSpec M p final := by
  rcases h with ⟨shape,off,on,values,sourceFrame,preparedFrame,entryFrame,xyFrame,doutFrame⟩
  refine ⟨shape,off,on,?_,sourceFrame,?_,?_,?_,?_⟩
  · intro ch enabled r q hr hq
    have value := congrArg (interpret M) (values ch enabled r q hr hq)
    simpa only [projectMemory,SwigluFwd.Whole.Tensor.read,entryGraph,source_graph_lowering] using value
  · intro allocation a outside
    exact congrArg (interpret M) (preparedFrame allocation a outside)
  · intro allocation a existed outside
    exact congrArg (interpret M) (entryFrame allocation a existed outside)
  · intro r q hr hq
    exact congrArg (interpret M) (xyFrame r q hr hq)
  · intro r q hr hq
    exact congrArg (interpret M) (doutFrame r q hr hq)

theorem host_outcomes (M : NativeModel α)
    (rt : Whole.Runtime (Term α) call (entryGraph entry) preexisting)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p) :
    HostSpec M rt (executeHost M rt) := by
  have original := Whole.host_outcomes graphAlgebra rt valid
  rcases original with failure | failure | failure | success
  · rcases failure with ⟨reason,hp,result⟩
    exact Or.inl ⟨reason,hp,by simp only [executeHost,result,projectOutcome]⟩
  · rcases failure with ⟨p,hp,hc,result⟩
    exact Or.inr (Or.inl ⟨p,hp,hc,by simp only [executeHost,result,projectOutcome]⟩)
  · rcases failure with ⟨p,reason,hp,hc,hs,result⟩
    exact Or.inr (Or.inr (Or.inl ⟨p,reason,hp,hc,hs,by simp only [executeHost,result,projectOutcome]⟩))
  · rcases success with ⟨p,plan,final,hp,hc,hs,result,spec⟩
    exact Or.inr (Or.inr (Or.inr ⟨p,plan,final,hp,hc,hs,
      by simp only [executeHost,result,projectOutcome],returned_projection M p final spec⟩))

#guard_msgs (drop info) in
#specNonCircular reference avoiding [interpret,Whole.scalar,Whole.sourceRun,Whole.executeHost,executeHost]
#axiomsClean host_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluBackward.Observed
