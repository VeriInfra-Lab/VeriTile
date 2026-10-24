import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.rotary_transform_ops.XorSource
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.TypedGrid
import bench.repairs.tritonbench_g.rotary_transform_ops.Spec
import VeriTile.Triton.Float.ReadProgramEquivalence

/-! Logical tensor channels for the complete original rotary grid. Source
FP32 storage and both signed metadata widths remain in the actual quotation.
The result view includes the copied suffix; only the rotary prefix is writable. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (s : Shape) : List IOChannel :=
  [⟨.real,[Host.size s],"x"⟩,
   ⟨.real,[s.coefficientRows*s.half],"cos"⟩,
   ⟨.real,[s.coefficientRows*s.half],"sin"⟩,
   ⟨.int,[s.batches+1],"cu"⟩,
   ⟨.int,[s.batches],"offsets"⟩,
   ⟨.real,[Host.size s],KernelAlgebra.regionName s.output⟩]
def outputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.batches,s.maximum,s.heads,s.width],KernelAlgebra.regionName s.output⟩]
abbrev Inputs (α : Type) (s : Shape) := ChannelValues α (inputs s)
abbrev Outputs (α : Type) (s : Shape) := ChannelValues α (outputs s)

def job {α : Type} (p : Ops α) (s : Shape) (cu64 off64 : Bool)
    (pid : Program) : Invocation α where
  kernel := TypedExecution.kernel s cu64 off64
  backend := KernelAlgebra.backend p
  pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2.1 else pid.2.2
  numPids := some (fun ax => if ax=0 then s.rowBlocks else if ax=1 then s.batches else s.heads)

def schedule {α : Type} (p : Ops α) (c : Config) (cu64 off64 : Bool) : List (Invocation α) :=
  if c.shape.total=0 ∨ c.shape.half=0 then []
  else (programs c).map (job p c.shape cu64 off64)

def io {α : Type} (p : Ops α) (c : Config) (cu64 off64 : Bool)
    (observe : α → α) : ReadProgramIO α α where
  inputs := inputs c.shape
  outputs := outputs c.shape
  scratch := []
  buffers := ["x","out","cos","sin","cu","offsets"]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun j => j.1.val
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _ => True
    | ⟨3,_⟩ => fun _ => c.shape.varlen=Bool.true
    | ⟨4,_⟩ => fun _ => c.shape.offsetTensor=Bool.true
    | ⟨5,_⟩ => fun j => 2*c.shape.half≤j.1.val%c.shape.width
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun j => c.shape.address (c.start j.1.val+j.2.1.val) j.2.2.1.val j.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.2.1.val<c.length j.1.val ∧ j.2.2.2.1.val<2*c.shape.half
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => schedule p c cu64 off64
  encode := id
  observe := fun _ _ => observe
  resultRead := fun port _ => match port with
    | ⟨0,_⟩ => fun j => c.shape.address (c.start j.1.val+j.2.1.val) j.2.2.1.val j.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  resultMask := fun port _ => match port with
    | ⟨0,_⟩ => fun j => j.2.1.val<c.length j.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

/-- Logical metadata and the existing out-of-place suffix-copy invariant.
In-place input/output aliasing is admitted by simultaneous channel binding. -/
def Ready {α : Type} (c : Config) (vals : Inputs α c.shape) : Prop :=
  (c.shape.varlen=Bool.true → ∀ b : Fin (c.shape.batches+1),
    vals (3 : Fin 6) (b,())=(c.metadata.cumulative b.val : Int)) ∧
  (c.shape.offsetTensor=Bool.true → ∀ b : Fin c.shape.batches,
    vals (4 : Fin 6) (b,())=(c.metadata.offsets b.val : Int)) ∧
  (∀ i : Fin (Host.size c.shape),2*c.shape.half ≤ i.val%c.shape.width →
    vals (5 : Fin 6) (i,())=vals (0 : Fin 6) (i,()))

/-- Independent logical payloads; no execution or mutable-memory state. -/
def inputMemory {α : Type} [Inhabited α] {s : Shape} (vals : Inputs α s) : Memory α :=
  fun r i => match r with
  | .input => if h : i<Host.size s then vals (0 : Fin 6) (⟨i,h⟩,()) else default
  | .cosine => if h : i<s.coefficientRows*s.half then vals (1 : Fin 6) (⟨i,h⟩,()) else default
  | .sine => if h : i<s.coefficientRows*s.half then vals (2 : Fin 6) (⟨i,h⟩,()) else default
  | _ => default

def realReference (c : Config) (vals : Inputs ℝ c.shape) : Outputs ℝ c.shape
  | ⟨0,_⟩ => fun j => realFormula c (inputMemory vals) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

local instance : Inhabited Native.Node := ⟨Native.atom Native.zero⟩

def observeNative (x : Native.Node) : Native.Node := Native.atom x.value
def nativeReference (c : Config) (p : Native.Instructions)
    (vals : Inputs Native.Node c.shape) : Outputs Native.Node c.shape
  | ⟨0,_⟩ => fun j => Native.atom (nativeFormula c p (Native.project (inputMemory vals))
      j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

/-- Independent logical input tensors and signed metadata. Entry output is
retained in full because suffix agreement is part of the original domain. -/
structure RotationInputs (α : Type) (s : Shape) where
  input : Fin s.total → Fin s.heads → Fin s.width → α
  cosine : Fin s.coefficientRows → Fin s.half → α
  sine : Fin s.coefficientRows → Fin s.half → α
  cumulative : Fin (s.batches+1) → Int
  offsets : Fin s.batches → Int
  entryOutput : Fin s.total → Fin s.heads → Fin s.width → α

def tensorIndex (s : Shape) : ((Fin s.total × Fin s.heads) × Fin s.width) ≃
    Fin (s.total*s.heads*s.width) :=
  (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv

def logicalInputs (c : Config) : Inputs α c.shape ≃ RotationInputs α c.shape where
  toFun := fun xs => {
    input := fun token head col => xs (0 : Fin 6) (tensorIndex c.shape ((token,head),col),())
    cosine := fun row pair => xs (1 : Fin 6) (finProdFinEquiv (row,pair),())
    sine := fun row pair => xs (2 : Fin 6) (finProdFinEquiv (row,pair),())
    cumulative := fun b => xs (3 : Fin 6) (b,())
    offsets := fun b => xs (4 : Fin 6) (b,())
    entryOutput := fun token head col => xs (5 : Fin 6) (tensorIndex c.shape ((token,head),col),()) }
  invFun := fun xs => Fin.cases
    (fun j => let coord := (tensorIndex c.shape).symm j.1
      xs.input coord.1.1 coord.1.2 coord.2)
    (Fin.cases (fun j => xs.cosine (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2)
      (Fin.cases (fun j => xs.sine (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2)
        (Fin.cases (fun j => xs.cumulative j.1) (Fin.cases (fun j => xs.offsets j.1)
          (Fin.cases (fun j => let coord := (tensorIndex c.shape).symm j.1
            xs.entryOutput coord.1.1 coord.1.2 coord.2) (fun i => i.elim0))))))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> apply congrArg (xs _) <;> apply Prod.ext
    all_goals first
      | exact (tensorIndex c.shape).apply_symm_apply j.1
      | exact finProdFinEquiv.apply_symm_apply j.1
      | rfl
  right_inv := by
    rintro ⟨x,co,si,cu,off,out⟩
    change RotationInputs.mk _ _ _ _ _ _ = RotationInputs.mk x co si cu off out
    congr 1
    · funext a b d
      exact congrArg (fun coord => x coord.1.1 coord.1.2 coord.2)
        ((tensorIndex c.shape).symm_apply_apply ((a,b),d))
    · funext a b
      exact congrArg (fun coord => co coord.1 coord.2) (finProdFinEquiv.symm_apply_apply (a,b))
    · funext a b
      exact congrArg (fun coord => si coord.1 coord.2) (finProdFinEquiv.symm_apply_apply (a,b))
    · funext a b d
      exact congrArg (fun coord => out coord.1.1 coord.1.2 coord.2)
        ((tensorIndex c.shape).symm_apply_apply ((a,b),d))

/-- Every result cell is a logical batch/row/head/column coordinate. -/
def logicalOutputs (c : Config) : Outputs α c.shape ≃
    (Fin c.shape.batches → Fin c.shape.maximum → Fin c.shape.heads → Fin c.shape.width → α) where
  toFun := fun ys batch row head col => ys (0 : Fin 1) (batch,row,head,col,())
  invFun := fun f => Fin.cases (fun j => f j.1 j.2.1 j.2.2.1 j.2.2.2.1) (fun i => i.elim0)
  left_inv := by
    intro ys
    funext port j
    have zero : port=0 := Fin.eq_zero port
    subst port
    exact congrArg (ys (0 : Fin 1)) (by cases j; rfl)
  right_inv := fun _ => rfl

/-- Original total extension, now backed by logical tensor coordinates. It
retains exactly the previous convention outside the finite input arrays. -/
def logicalMemory [Inhabited α] (c : Config) (xs : RotationInputs α c.shape) : Memory α :=
  inputMemory ((logicalInputs c).symm xs)

def RotationInputs.inputAt [Inhabited α] (c : Config) (xs : RotationInputs α c.shape)
    (token head col : Nat) : α := logicalMemory c xs .input (c.shape.address token head col)
def RotationInputs.cosineAt [Inhabited α] (c : Config) (xs : RotationInputs α c.shape)
    (row pair : Nat) : α := logicalMemory c xs .cosine (row*c.shape.half+pair)
def RotationInputs.sineAt [Inhabited α] (c : Config) (xs : RotationInputs α c.shape)
    (row pair : Nat) : α := logicalMemory c xs .sine (row*c.shape.half+pair)


/-- Logical tensor coordinates for the exact original readback judgment. The
result window includes the suffix while the writable frame remains the prefix. -/
def logicalIO (c : Config) (cu64 off64 : Bool) :
    LogicalDomainIO (Inputs ℝ c.shape) (Outputs ℝ c.shape) (RotationInputs ℝ c.shape)
      (Fin c.shape.batches → Fin c.shape.maximum → Fin c.shape.heads → Fin c.shape.width → ℝ)
      (io realOps c cu64 off64 id).ImplementsOn where
  inputs := logicalInputs c
  outputs := logicalOutputs c

/-- The original signed metadata and prepared-suffix domain, transported without
adding a caller condition or discarding any entry-output values. -/
def LogicalReady (c : Config) (xs : RotationInputs ℝ c.shape) : Prop :=
  Ready c ((logicalInputs c).symm xs)

end
end VeriTile.Bench.Repairs.RotaryTransformOps.Interface

namespace VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
open VeriTile.Triton FP.Structural
noncomputable section

/-- Candidate uses the same complete geometry and metadata storage selection. -/
def kernel (s : Shape) (cumulative64 offsets64 : Bool) : ComputeKernel :=
  XorSource.source cumulative64 offsets64 (TypedExecution.outputRegion s) "x" "cos" "sin" "cu" "offsets"
    s.scalarOffset s.maximum (2*s.half) s.coefficientRows
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    s.blockK s.blockM s.offsetTensor s.varlen s.interleaved s.conjugate

variable {α : Type} [Inhabited α]

def job (ops : Ops α) (s : Shape) (cu64 off64 : Bool) (pid : Program) : Invocation α :=
  { Interface.job ops s cu64 off64 pid with kernel := kernel s cu64 off64 }

def schedule (ops : Ops α) (c : Config) (cu64 off64 : Bool) : List (Invocation α) :=
  if c.shape.total=0 ∨ c.shape.half=0 then []
  else (programs c).map (job ops c.shape cu64 off64)

def variant (ops : Ops α) (c : Config) (cu64 off64 : Bool) (observe : α → α) :
    ReadProgramIO.Variant (Interface.io ops c cu64 off64 observe) where
  schedule := fun _ => schedule ops c cu64 off64
  observe := fun _ _ => observe

end
end VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
