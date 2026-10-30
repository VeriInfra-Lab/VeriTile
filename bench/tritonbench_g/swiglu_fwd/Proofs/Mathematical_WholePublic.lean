import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholeHost

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Paired SwiGLU has a positive even final dimension; zero-size batch axes
remain allowed. This is a public mathematical input condition, not a source
assertion or a manufactured error on odd-width calls. -/
def InputDomain (input : Tensor) : Prop :=
  ∃ batch : List Nat, ∃ ncols : Nat, 0 < ncols ∧ input.shape = batch ++ [2*ncols]

/-- Pure mathematical interpretation; this is explicitly not an IEEE backend. -/
def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .mul => x*y | .add => x+y | .sub => x-y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .sigmoid => Real.sigmoid x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => 0

theorem scalar_real (x y : ℝ) : scalar realAlgebra x y = Math.forward x y := by
  rw [Math.forward_eq]
  rfl

/-- Public source operation order, excluding ordinary internal Torch/runtime
steps. Source text/AST hashes and CPU metadata probes bind this list to Python. -/
def hostOperations (input : Tensor) (supplied : Option Tensor) : List String :=
  ["xy.stride(-1)"] ++ (if input.lastStride ≠ 1 then ["xy.contiguous()"] else []) ++
  ["xy.shape[:-1]","xy.reshape(-1,xy.shape[-1])","xy.chunk(2,dim=-1)"] ++
  (if supplied.isSome then ["out.reshape(-1,out.shape[-1])","assert out.shape==x.shape"]
   else ["torch.empty_like(x)"]) ++
  ["assert out.stride(-1)==1","M,N=x.shape","grid=(M,cdiv(N,BLOCK_N))",
   "torch.cuda.device(x.device.index)","autotune key=[ncols]","selected kernel launch",
   "out.reshape(*batch_shape,out.shape[-1])"]

structure Plan (p : Prepared α input supplied entry preexisting) where
  trials : List (Launch p.layout)
  selected : Launch p.layout
  device : Nat
  deviceBinding : device = input.cudaDevice
  key : List Nat
  keyBinding : key = [p.ncols]

/-- Preparation contains ordinary view/copy/allocator contracts only. Setup
may fail; on success it chooses a device, key, configurations and schedules.
Neither callback supplies the final kernel output. -/
structure Runtime (α : Type) (input : Tensor) (supplied : Option Tensor) (entry : PhysicalMemory α)
    (preexisting : Nat → Prop) where
  prepare : Except String (Prepared α input supplied entry preexisting)
  setup : (p : Prepared α input supplied entry preexisting) → Except String (Plan p)

inductive Outcome (α : Type) where
  | preparationFailure (reason : String)
  | assertionFailure
  | runtimeFailure (reason : String)
  | kernelFailure
  | returned (shape : List Nat) (allocation : Nat) (base rowStride : Nat)
      (memory : PhysicalMemory α)

def executeHost [Inhabited α] (M : Algebra α) (rt : Runtime α input supplied entry preexisting) : Outcome α :=
  match rt.prepare with
  | .error reason => .preparationFailure reason
  | .ok p =>
    if checks p then
      match rt.setup p with
      | .error reason => .runtimeFailure reason
      | .ok plan =>
        match entryRun M p plan.trials plan.selected with
        | none => .kernelFailure
        | some final => .returned p.returnShape p.out.allocation p.out.base p.out.rowStride (p.commit final)
    else .assertionFailure

/-- Independent scalar contract lifted to the original logical tensor halves.
`formula` is independent of execution and receives values at entry. -/
def ReturnedSpec [Inhabited α] (formula : α → α → α)
    (p : Prepared α input supplied entry preexisting) (memory : PhysicalMemory α) : Prop :=
  p.returnShape = p.batchShape ++ [p.ncols] ∧
  (∀ r q, r < p.batchShape.prod → q < p.ncols →
    memory p.out.allocation (p.out.base+r*p.out.rowStride+q) =
      formula (input.read entry (r*(2*p.ncols)+q))
        (input.read entry (r*(2*p.ncols)+(p.ncols+q)))) ∧
  (∀ allocation a, PhysicalOutside p allocation a → memory allocation a = p.memory allocation a) ∧
  (∀ allocation a, preexisting allocation → PhysicalOutside p allocation a →
    memory allocation a = entry allocation a) ∧
  (∀ k, PhysicalOutside p input.allocation (input.offset k) →
    input.read memory k = input.read entry k) ∧
  (∀ r q, r < p.batchShape.prod → q < 2*p.ncols →
    p.flat.read memory r q = p.flat.read p.memory r q)

def HostSpec [Inhabited α] (formula : α → α → α)
    (rt : Runtime α input supplied entry preexisting) (outcome : Outcome α) : Prop :=
  (∃ reason, rt.prepare = .error reason ∧ outcome = .preparationFailure reason) ∨
  (∃ p, rt.prepare = .ok p ∧ checks p = Bool.false ∧ outcome = .assertionFailure) ∨
  (∃ p reason, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .error reason ∧
    outcome = .runtimeFailure reason) ∨
  (∃ p plan final, rt.prepare = .ok p ∧ checks p = Bool.true ∧ rt.setup p = .ok plan ∧
    outcome = .returned p.returnShape p.out.allocation p.out.base p.out.rowStride (p.commit final) ∧
    ReturnedSpec formula p (p.commit final))

/-- Successful source execution and ordinary failure observations of the full
entry. Public validity concerns bounds, pairing and ownership; it does not
contain a final-state equality or a per-kernel result premise. -/
theorem host_outcomes [Inhabited α] (M : Algebra α) (rt : Runtime α input supplied entry preexisting)
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
        obtain ⟨final,ran,shape,values,frame⟩ := entry_result M p d plan.trials plan.selected
        apply Or.inr; apply Or.inr; apply Or.inr
        refine ⟨p,plan,final,hp,hc,hs,?_,shape,values,?_,?_,?_,?_⟩
        · simp [executeHost,hp,hc,hs,ran]
        · exact commit_frame p final frame
        · exact entry_memory_frame p final frame
        · exact original_input_frame p final frame
        · exact live_input_preserved p d final frame

theorem real_host_outcomes (rt : Runtime ℝ input supplied entry preexisting)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p) :
    HostSpec Math.forward rt (executeHost realAlgebra rt) := by
  have h := host_outcomes realAlgebra rt valid
  have scalarEq : scalar realAlgebra = Math.forward := by funext x y; exact scalar_real x y
  simpa only [scalarEq] using h

#guard_msgs (drop info) in
#specNonCircular scalar avoiding [Layout.source,sourceRun,tileExec,runTiles,entryRun,executeHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
