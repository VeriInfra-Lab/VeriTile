import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.rotary_transform.Proofs.TypedGrid
import bench.tritonbench_g.rotary_transform.Proofs.Spec
import VeriTile.Triton.Float.ReadProgramIO

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def dataSize (s : Shape) := s.total*s.heads*s.width
def coefficientSize (s : Shape) := s.coefficientRows*s.half

def inputs (c : Config) : List IOChannel :=
  [⟨.real,[dataSize c.shape],"x"⟩,
   ⟨.real,[coefficientSize c.shape],"cos"⟩,
   ⟨.real,[coefficientSize c.shape],"sin"⟩,
   ⟨.int,[c.shape.batches+1],"cu"⟩,
   ⟨.int,[c.shape.batches],"offsets"⟩,
   ⟨.real,[dataSize c.shape],TypedExecution.outputRegion c.shape⟩]

def outputs (c : Config) : List IOChannel :=
  [⟨.real,[c.shape.batches,c.shape.maximum,c.shape.heads,c.shape.width],
    TypedExecution.outputRegion c.shape⟩]

def invocation (p : Ops α) (c : Config) (cumulative64 offsets64 : Bool)
    (pid : Program) : Invocation α where
  kernel := TypedExecution.kernel c.shape cumulative64 offsets64
  backend := KernelAlgebra.backend p
  pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2.1 else pid.2.2
  numPids := some (fun ax => if ax=0 then c.shape.rowBlocks else
    if ax=1 then c.shape.batches else c.shape.heads)

def io (p : Ops α) (observe : α → α) (c : Config) (cumulative64 offsets64 : Bool) :
    ReadProgramIO α α where
  inputs := inputs c
  outputs := outputs c
  scratch := []
  buffers := ["x","cos","sin","cu","offsets",TypedExecution.outputRegion c.shape]
  read := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun lane => lane.1.val
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  inputMask := fun port _ => match port with
    | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _ => True
    | ⟨3,_⟩ => fun _ => c.shape.varlen = Bool.true
    | ⟨4,_⟩ => fun _ => c.shape.offsetTensor = Bool.true
    | ⟨5,_⟩ => fun lane => 2*c.shape.half ≤ lane.1.val%c.shape.width
    | ⟨n+6,h⟩ => False.elim (by simp [inputs] at h)
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => c.shape.address (c.start lane.1.val+lane.2.1.val)
      lane.2.2.1.val lane.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  outputMask := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val<c.length lane.1.val ∧
      lane.2.2.2.1.val<2*c.shape.half
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => if c.shape.total=0 ∨ c.shape.half=0 then [] else
    (programs c).map (invocation p c cumulative64 offsets64)
  encode := id
  observe := fun _ _ => observe
  resultRead := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => c.shape.address (c.start lane.1.val+lane.2.1.val)
      lane.2.2.1.val lane.2.2.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)
  resultMask := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val<c.length lane.1.val
    | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

/-- Integer metadata equals the logical configuration. The already allocated
output suffix contains the corresponding input values; for inplace this is
the same input channel and introduces no additional layout restriction. -/
def domain (c : Config) (vals : ChannelValues α (inputs c)) : Prop :=
  (c.shape.varlen = Bool.true → ∀ i : Fin (c.shape.batches+1),
    vals ⟨3,by simp [inputs]⟩ (i,()) = (c.metadata.cumulative i.val : Int)) ∧
  (c.shape.offsetTensor = Bool.true → ∀ i : Fin c.shape.batches,
    vals ⟨4,by simp [inputs]⟩ (i,()) = (c.metadata.offsets i.val : Int)) ∧
  (∀ i : Fin (dataSize c.shape),2*c.shape.half ≤ i.val%c.shape.width →
    vals ⟨5,by simp [inputs]⟩ (i,()) = vals ⟨0,by simp [inputs]⟩ (i,()))

def arrayMemory (c : Config) (vals : ChannelValues α (inputs c)) : Memory α :=
  fun role off => match role with
    | .input => if h : off<dataSize c.shape then vals ⟨0,by simp [inputs]⟩ (⟨off,h⟩,()) else default
    | .cosine => if h : off<coefficientSize c.shape then vals ⟨1,by simp [inputs]⟩ (⟨off,h⟩,()) else default
    | .sine => if h : off<coefficientSize c.shape then vals ⟨2,by simp [inputs]⟩ (⟨off,h⟩,()) else default
    | _ => default

def expected (c : Config) (f : Memory α → Nat → Nat → Nat → Nat → α)
    (vals : ChannelValues α (inputs c)) : ChannelValues α (outputs c) :=
  fun port => match port with
    | ⟨0,_⟩ => fun lane => f (arrayMemory c vals) lane.1.val lane.2.1.val
      lane.2.2.1.val lane.2.2.2.1.val
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

def logicalInputs (c : Config) : ChannelValues α (inputs c) ≃ RotationInputs α c.shape where
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
def logicalOutputs (c : Config) : ChannelValues α (outputs c) ≃
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
def logicalMemory (c : Config) (xs : RotationInputs α c.shape) : Memory α :=
  arrayMemory c ((logicalInputs c).symm xs)

def RotationInputs.inputAt (c : Config) (xs : RotationInputs α c.shape)
    (token head col : Nat) : α := logicalMemory c xs .input (c.shape.address token head col)
def RotationInputs.cosineAt (c : Config) (xs : RotationInputs α c.shape)
    (row pair : Nat) : α := logicalMemory c xs .cosine (row*c.shape.half+pair)
def RotationInputs.sineAt (c : Config) (xs : RotationInputs α c.shape)
    (row pair : Nat) : α := logicalMemory c xs .sine (row*c.shape.half+pair)


/-- Logical tensor coordinates for the exact original readback judgment. The
result window includes the suffix while the writable frame remains the prefix. -/
def logicalIO (c : Config) (cu64 off64 : Bool) :
    LogicalDomainIO (ChannelValues ℝ (inputs c)) (ChannelValues ℝ (outputs c)) (RotationInputs ℝ c.shape)
      (Fin c.shape.batches → Fin c.shape.maximum → Fin c.shape.heads → Fin c.shape.width → ℝ)
      (io realOps id c cu64 off64).ImplementsOn where
  inputs := logicalInputs c
  outputs := logicalOutputs c

/-- The original signed metadata and prepared-suffix domain, transported without
adding a caller condition or discarding any entry-output values. -/
def LogicalReady (c : Config) (xs : RotationInputs ℝ c.shape) : Prop :=
  domain c ((logicalInputs c).symm xs)

end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
