import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeHost
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeMath
import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholePublic

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Paired real SwiGLU: positive even last dimension, arbitrary batch axes
including empty batches. The source's own dout/dxy/out assertions are retained. -/
def InputDomain (call : Call) : Prop :=
  ∃ batch : List Nat, ∃ ncols : Nat, 0 < ncols ∧ call.input.shape = batch ++ [2*ncols]

def formula : Channel → ℝ → ℝ → ℝ → ℝ
  | .dx => Math.dx
  | .dy => fun x _ d => Math.dy x d
  | .out => fun x y _ => Math.forward x y

abbrev realAlgebra := SwigluFwd.Whole.realAlgebra

theorem scalar_real (ch : Channel) (x y d : ℝ) :
    scalar realAlgebra ch x y d = formula ch x y d := by
  cases ch <;> simp only [scalar,formula,realAlgebra,SwigluFwd.Whole.realAlgebra,
    Math.dx,Math.dy,Math.forward,Math.sigmoid_eq] <;> ring

/-- Every source operation, in order; branch-dependent allocations, asserts
and both return alternatives are kept. Runtime-internal steps are abstract. -/
def hostOperations (call : Call) : List String :=
  ["xy.stride(-1)"] ++ (if call.input.lastStride ≠ 1 then ["xy.contiguous()"] else []) ++
  ["dout.stride(-1)"] ++ (if call.upstream.lastStride ≠ 1 then ["dout.contiguous()"] else []) ++
  ["batch_shape=xy.shape[:-1]","xy.reshape(-1,xy.shape[-1])","x,y=xy.chunk(2,dim=-1)",
   "dout.reshape(-1,dout.shape[-1])","assert dout.shape==x.shape"] ++
  (if call.gradient.isSome then ["dxy.reshape(-1,dxy.shape[-1])","assert dxy.shape==xy.shape"]
   else ["dxy=torch.empty_like(xy)"]) ++
  ["dx,dy=dxy.chunk(2,dim=-1)","assert dx.stride(-1)==1","assert dy.stride(-1)==1"] ++
  (if call.recompute then
    (if call.output.isSome then ["out.reshape(-1,out.shape[-1])","assert out.shape==x.shape"]
     else ["out=torch.empty_like(x)"]) ++ ["assert out.stride(-1)==1"] else []) ++
  ["M,N=x.shape","grid=(M,cdiv(N,BLOCK_N))","torch.cuda.device(x.device.index)",
   "OUT=out if recompute_output else None","RECOMPUTE_OUTPUT=(OUT is not None)",
   "autotune configs=[32,64,128,256,512,1024], key=[ncols]","all trial launches",
   "selected kernel launch","dxy.reshape(*batch_shape,dxy.shape[-1])"] ++
  (if call.recompute then ["out.reshape(*batch_shape,out.shape[-1])","return (dxy,out)"]
   else ["return dxy"])

structure Plan (p : Prepared α call entry preexisting) where
  trials : List (Launch p.layout)
  selected : Launch p.layout
  device : Nat
  deviceBinding : device = call.input.cudaDevice
  key : List Nat
  keyBinding : key = [p.ncols]

/-- Ordinary Torch preparation and runtime choices. No callback computes a
kernel result. Errors are observations from these ordinary runtime boundaries. -/
structure Runtime (α : Type) (call : Call) (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  prepare : Except String (Prepared α call entry preexisting)
  setup : (p : Prepared α call entry preexisting) → Except String (Plan p)

structure ReturnedTensor where
  shape : List Nat
  allocation : Nat
  base : Nat
  rowStride : Nat
  deriving DecidableEq

def Prepared.returnGradient (p : Prepared α call entry preexisting) : ReturnedTensor :=
  ⟨p.batchShape++[p.grad.matrix.cols],p.grad.matrix.allocation,p.grad.matrix.base,p.grad.matrix.rowStride⟩

def Prepared.returnOutput (p : Prepared α call entry preexisting) : Option ReturnedTensor :=
  if call.recompute then some ⟨p.batchShape++[p.out.matrix.cols],p.out.matrix.allocation,
    p.out.matrix.base,p.out.matrix.rowStride⟩ else none

inductive Outcome (α : Type) where
  | preparationFailure (reason : String)
  | assertionFailure
  | runtimeFailure (reason : String)
  | kernelFailure
  | returned (gradient : ReturnedTensor) (output : Option ReturnedTensor) (memory : PhysicalMemory α)

def executeHost [Inhabited α] (M : Algebra α) (rt : Runtime α call entry preexisting) : Outcome α :=
  match rt.prepare with
  | .error reason => .preparationFailure reason
  | .ok p => if checks p then
      match rt.setup p with
      | .error reason => .runtimeFailure reason
      | .ok plan => match entryRun M p plan.trials plan.selected with
        | none => .kernelFailure
        | some final => .returned p.returnGradient p.returnOutput (p.commit final)
    else .assertionFailure

def ReturnedSpec [Inhabited α] (spec : Channel → α → α → α → α)
    (p : Prepared α call entry preexisting) (final : State α) : Prop :=
  p.returnGradient.shape = call.input.shape ∧
  (call.recompute = Bool.false → p.returnOutput = none) ∧
  (call.recompute = Bool.true → ∃ out, p.returnOutput = some out ∧
    out.shape = p.batchShape++[p.ncols]) ∧
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.commit final (p.location ch r q).1 (p.location ch r q).2 = spec ch
      (call.input.read entry (r*(2*p.ncols)+q))
      (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
      (call.upstream.read entry (r*p.ncols+q))) ∧
  (∀ region a, WholeOutside p.layout call.recompute region a →
    final.mem region a = p.state.mem region a) ∧
  (∀ allocation a, PhysicalOutside p allocation a → p.commit final allocation a = p.memory allocation a) ∧
  (∀ allocation a, preexisting allocation → PhysicalOutside p allocation a →
    p.commit final allocation a = entry allocation a) ∧
  (∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.xy.matrix.read (p.commit final) r q = p.xy.matrix.read p.memory r q) ∧
  (∀ r q, r < p.batchShape.prod → q < p.ncols →
    p.dout.matrix.read (p.commit final) r q = p.dout.matrix.read p.memory r q)

def HostSpec [Inhabited α] (spec : Channel → α → α → α → α)
    (rt : Runtime α call entry preexisting) (outcome : Outcome α) : Prop :=
  (∃ reason, rt.prepare = .error reason ∧ outcome = .preparationFailure reason) ∨
  (∃ p, rt.prepare = .ok p ∧ checks p = Bool.false ∧ outcome = .assertionFailure) ∨
  (∃ p reason, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .error reason ∧
    outcome = .runtimeFailure reason) ∨
  (∃ p plan final, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .ok plan ∧
    outcome = .returned p.returnGradient p.returnOutput (p.commit final) ∧ ReturnedSpec spec p final)

theorem host_outcomes [Inhabited α] (M : Algebra α) (rt : Runtime α call entry preexisting)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p) :
    HostSpec (scalar M) rt (executeHost M rt) := by
  cases hp : rt.prepare with
  | error reason => exact Or.inl ⟨reason,hp,by simp [executeHost,hp]⟩
  | ok p =>
    cases hc : checks p with
    | «false» => exact Or.inr (Or.inl ⟨p,hp,hc,by simp [executeHost,hp,hc]⟩)
    | «true» =>
      cases hs : rt.setup p with
      | error reason => exact Or.inr (Or.inr (Or.inl ⟨p,reason,hp,hc,hs,by simp [executeHost,hp,hc,hs]⟩))
      | ok plan =>
        let d := valid p hp hc
        obtain ⟨final,ran,values,frame⟩ := entry_result M p d plan.trials plan.selected
        apply Or.inr; apply Or.inr; apply Or.inr
        refine ⟨p,plan,final,hp,hc,hs,?_,?_,?_,?_,values,frame,
          commit_frame p final,entry_frame p final,?_⟩
        · simp [executeHost,hp,hc,hs,ran]
        · simp only [Prepared.returnGradient,d.gradientShape.2,p.sourceShape]
        · intro off; simp [Prepared.returnOutput,off]
        · intro on
          refine ⟨⟨p.batchShape++[p.out.matrix.cols],p.out.matrix.allocation,p.out.matrix.base,p.out.matrix.rowStride⟩,?_,?_⟩
          · simp [Prepared.returnOutput,on]
          · simp [(d.outputShape on).2]
        · exact live_inputs_preserved p d final

theorem real_host_outcomes (rt : Runtime ℝ call entry preexisting)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p) :
    HostSpec formula rt (executeHost realAlgebra rt) := by
  have h := host_outcomes realAlgebra rt valid
  have eq : scalar realAlgebra = formula := by funext ch x y d; exact scalar_real ch x y d
  simpa only [eq] using h

#guard_msgs (drop info) in
#specNonCircular formula avoiding [Layout.source,sourceRun,tileExec,runTiles,entryRun,executeHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
