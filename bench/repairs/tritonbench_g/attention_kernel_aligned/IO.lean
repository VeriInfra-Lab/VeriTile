import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Domain
import VeriTile.Triton.Float.ObservedProgramIO
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.KernelInterface

/-! Finite tensor ports. Q/K/V use their original four-dimensional strided
windows; bias uses its original 128 stored entries per row. Read/read aliases
are admitted by simultaneous input bindings. No padded or exterior cell is
included in an input window. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
open VeriTile Triton FP.Structural Full Whole
noncomputable section

def inputRegion (p : Parameters) : Fin 4 → RegionName := ![p.q,p.k,p.v,p.bias]
def inputStrides (p : Parameters) : Fin 4 → Strides := ![qStrides p,kStrides p,vStrides p,bStrides p]
def inputWidth (p : Parameters) (role : Fin 4) : Nat := if role.val=3 then 128 else p.head
def shape (p : Parameters) (width : Nat) : TileShape := [p.batches,p.heads,p.length,width]

structure Point (p : Parameters) where
  role : Fin 4
  batch : Fin p.batches
  head : Fin p.heads
  row : Fin p.length
  col : Fin (inputWidth p role)

def pointAddress (p : Parameters) (i : Point p) : Nat :=
  offset (inputStrides p i.role) i.batch.val i.head.val i.row.val i.col.val

def channels (dtype : TileDType) (p : Parameters) : List IOChannel :=
  [⟨dtype,shape p p.head,p.q⟩,⟨dtype,shape p p.head,p.k⟩,
   ⟨dtype,shape p p.head,p.v⟩,⟨dtype,shape p 128,p.bias⟩]

def point (dtype : TileDType) (p : Parameters) : (i : Fin (channels dtype p).length) →
    TileIndex (channels dtype p)[i].shape → Point p :=
  Fin.cases (fun j => ⟨0,j.1,j.2.1,j.2.2.1,j.2.2.2.1⟩)
    (Fin.cases (fun j => ⟨1,j.1,j.2.1,j.2.2.1,j.2.2.2.1⟩)
      (Fin.cases (fun j => ⟨2,j.1,j.2.1,j.2.2.1,j.2.2.2.1⟩)
        (Fin.cases (fun j => ⟨3,j.1,j.2.1,j.2.2.1,j.2.2.2.1⟩) (fun i => i.elim0))))

def pointValue {α : Type} (dtype : TileDType) (p : Parameters) (v : ChannelValues α (channels dtype p)) (i : Point p) : Value α dtype :=
  let f : (role : Fin 4) → Fin (inputWidth p role) → Value α dtype :=
    Fin.cases (fun d => v (0 : Fin 4) (i.batch,i.head,i.row,d,PUnit.unit))
      (Fin.cases (fun d => v (1 : Fin 4) (i.batch,i.head,i.row,d,PUnit.unit))
        (Fin.cases (fun d => v (2 : Fin 4) (i.batch,i.head,i.row,d,PUnit.unit))
          (Fin.cases (fun d => v (3 : Fin 4) (i.batch,i.head,i.row,d,PUnit.unit)) (fun k => k.elim0))))
  f i.role i.col

/-- Total extension of the finite input ports. At aliased coordinates any
witness gives the same bound input payload. Outside the declared windows the
extension is arbitrary and is never read by the mathematical/native target. -/
def readInputs {α : Type} [Inhabited α] (dtype : TileDType) (p : Parameters) (v : ChannelValues α (channels dtype p))
    (r : RegionName) (off : Nat) : Value α dtype := by
  classical
  exact if h : ∃ i : Point p,inputRegion p i.role=r ∧ pointAddress p i=off
    then pointValue dtype p v h.choose else defaultValue dtype

def layout (α : Type) (dtype : TileDType) (p : Parameters) : MemoryIO α where
  inputs := channels dtype p
  outputs := [⟨dtype,shape p p.head,p.output⟩]
  scratch := []
  buffers := [p.q,p.k,p.v,p.bias,p.output]
  read := fun i _ j => pointAddress p (point dtype p i j)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => offset (oStrides p) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
    (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Interface

/-! Actual source schedules and independent logical targets for both original
storage branches. The entire supplied grid permutation is retained. Native
inputs retain arbitrary symbolic payloads; observation alone produces words. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
open VeriTile Triton FP.Structural Full Whole FPValue
noncomputable section

def job {α : Type} (M : Algebra α) (P : Primitives α) (p : Parameters) (i : Program) : Invocation α where
  kernel := Source.source p
  backend := backend M P
  clearRegisters := Bool.true
  pids := ids i
  numPids := some (dimensions (grid p))

def jobs {α : Type} (M : Algebra α) (P : Primitives α) (p : Parameters) (work : List Program) : List (Invocation α) :=
  work.map (job M P p)

def realIO (p : Parameters) (work : List Program) : ProgramIO ℝ Real.R where
  toMemoryIO := layout ℝ .fp16 p
  schedule := fun _ => jobs Real.algebra Real.model p work
  encode := some
  observe := fun _ _ x => x.unbotD 0

def realFormula (p : Parameters) (work : List Program) (v : (realIO p work).Inputs) : (realIO p work).Outputs :=
  Fin.cases (fun j => Real.expected p (readInputs .fp16 p v) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
    (fun i => i.elim0)

def nativeIO (p : Parameters) (work : List Program) : ProgramIO Native.Expr Native.Expr where
  toMemoryIO := layout Native.Expr .fp16 p
  schedule := fun _ => jobs (Native.algebraFor p) (Native.modelFor p) p work
  encode := id
  observe := fun _ _ => id

def nativeObserver (I : Native.Instructions) (p : Parameters) (work : List Program) :
    (o : Fin (nativeIO p work).outputs.length) → TileIndex (nativeIO p work).outputs[o].shape → Native.Expr → BitVec 16 :=
  Fin.cases (fun j => RawKernel.nativeDecode I j.2.2.2.1.val) (fun i => i.elim0)

def nativeFormula (I : Native.Instructions) (p : Parameters) (selected : Selected p)
    (work : List Program) (v : (nativeIO p work).Inputs) : (nativeIO p work).ObservedOutputs (BitVec 16) :=
  Fin.cases (fun j => RawKernel.nativeDecode I j.2.2.2.1.val
    (KernelOnly.Specification.outputValue (Native.modelFor p) p (readInputs .fp16 p v)
      (j.2.2.1.val/p.blockM) j.1.val j.2.1.val
      (lane p selected j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt))) (fun i => i.elim0)

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Interface

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Interface
open VeriTile Triton FP.Structural Full Whole FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned.Interface (layout readInputs)
noncomputable section

def job {α : Type} (M : Algebra α) (P : Primitives α) (p : Parameters) (i : Program) : Invocation α where
  kernel := Source.source p
  backend := backend M P
  clearRegisters := Bool.true
  pids := ids i
  numPids := some (dimensions (grid p))

def jobs {α : Type} (M : Algebra α) (P : Primitives α) (p : Parameters) (work : List Program) : List (Invocation α) :=
  work.map (job M P p)

def realIO (p : Parameters) (work : List Program) : ProgramIO ℝ Real.R where
  toMemoryIO := layout ℝ .bf16 p
  schedule := fun _ => jobs Real.algebra Real.model p work
  encode := some
  observe := fun _ _ x => x.unbotD 0

def realFormula (p : Parameters) (work : List Program) (v : (realIO p work).Inputs) : (realIO p work).Outputs :=
  Fin.cases (fun j => Real.expected p (readInputs .bf16 p v) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
    (fun i => i.elim0)

def nativeIO (p : Parameters) (work : List Program) : ProgramIO Native.Expr Native.Expr where
  toMemoryIO := layout Native.Expr .bf16 p
  schedule := fun _ => jobs (Native.algebraFor p) (Native.modelFor p) p work
  encode := id
  observe := fun _ _ => id

def nativeObserver (I : Native.Instructions) (p : Parameters) (work : List Program) :
    (o : Fin (nativeIO p work).outputs.length) → TileIndex (nativeIO p work).outputs[o].shape → Native.Expr → BitVec 16 :=
  Fin.cases (fun j => RawKernel.nativeDecode I j.2.2.2.1.val) (fun i => i.elim0)

def nativeFormula (I : Native.Instructions) (p : Parameters) (selected : Selected p)
    (work : List Program) (v : (nativeIO p work).Inputs) : (nativeIO p work).ObservedOutputs (BitVec 16) :=
  Fin.cases (fun j => RawKernel.nativeDecode I j.2.2.2.1.val
    (KernelOnly.Specification.outputValue (Native.modelFor p) p (readInputs .bf16 p v)
      (j.2.2.1.val/p.blockM) j.1.val j.2.1.val
      (lane p selected j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt))) (fun i => i.elim0)

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Interface

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Pair
open VeriTile Triton FP.Structural Full Whole FPValue Interface
noncomputable section

def job (p : Parameters) (pid : Program) : Invocation Native.Expr :=
  {Interface.job (Native.algebraFor p) (Native.modelFor p) p pid with kernel := Source.queryEarly p}

def variant (p : Parameters) (work : List Program) : ProgramIO.Variant (nativeIO p work) where
  schedule := fun _ => work.map (job p)
  observe := (nativeIO p work).observe

def original (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.original (nativeIO p work) (fun _ => True)

def queryEarly (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.ofVariant (variant p work) (fun _ => True)

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Pair

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Pair
open VeriTile Triton FP.Structural Full Whole FPValue Interface
noncomputable section

def job (p : Parameters) (pid : Program) : Invocation Native.Expr :=
  {Interface.job (Native.algebraFor p) (Native.modelFor p) p pid with kernel := Source.queryEarly p}

def variant (p : Parameters) (work : List Program) : ProgramIO.Variant (nativeIO p work) where
  schedule := fun _ => work.map (job p)
  observe := (nativeIO p work).observe

def original (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.original (nativeIO p work) (fun _ => True)

def queryEarly (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.ofVariant (variant p work) (fun _ => True)

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Pair

/-! Logical array coordinates for the existing complete IO. The storage field
retains every original input cell, including aliases and padding; the view is
an equivalence and never replaces the underlying source, schedule or frame. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.LogicalIO
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

structure Backing (p : Parameters) where
  storage : ChannelValues ℝ (Interface.layout ℝ .fp16 p).inputs

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (p.head) → ℝ

def outputsEquiv (p : Parameters) (work : List Program) :
    (Interface.realIO p work).Outputs ≃ Outputs p where
  toFun := fun values b h row col => values (0 : Fin 1) (b,h,row,col,())
  invFun := fun values => Fin.cases
    (fun j => values j.1 j.2.1 j.2.2.1 j.2.2.2.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro values
    funext o j
    have zero : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,h,row,col,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl


def read (xs : Backing p) := Interface.readInputs .fp16 p xs.storage

def Backing.query (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.q (b*p.qBatchStride+h*p.qHeadStride+m*p.qRowStride+e*p.qColStride)

def Backing.key (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.k (b*p.kBatchStride+h*p.kHeadStride+e*p.kDimStride+m*p.kSeqStride)

def Backing.value (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.v (b*p.vBatchStride+h*p.vHeadStride+m*p.vSeqStride+e*p.vDimStride)

def Backing.bias (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.bias (b*p.biasBatchStride+h*p.biasHeadStride+m*p.biasRowStride+e)

/-- Mathematical input arrays in logical coordinates. Total extensions outside
finite windows retain the prior specification's exact alias convention. -/
structure Arrays (p : Parameters) where
  query : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  key : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  value : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  bias : Fin p.batches → Fin p.heads → Nat → Nat → ℝ

def arraysOfBacking (p : Parameters) (xs : Backing p) : Arrays p where
  query := xs.query
  key := xs.key
  value := xs.value
  bias := xs.bias

/-- The logical arrays and complete original windows are tied by an equality.
This keeps aliases, padding and unused cells without treating them as new
independent mathematical inputs. -/
structure Inputs (p : Parameters) where
  arrays : Arrays p
  backing : Backing p
  connection : arrays = arraysOfBacking p backing

def Inputs.query (xs : Inputs p) := xs.arrays.query
def Inputs.key (xs : Inputs p) := xs.arrays.key
def Inputs.value (xs : Inputs p) := xs.arrays.value
def Inputs.bias (xs : Inputs p) := xs.arrays.bias

def inputsEquiv (p : Parameters) (work : List Program) :
    (Interface.realIO p work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO ℝ Real.R (Inputs p) (Outputs p) where
  core := Interface.realIO p work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.LogicalIO

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.LogicalIO
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

structure Backing (p : Parameters) where
  storage : ChannelValues ℝ (VeriTile.Bench.Repairs.AttentionKernelAligned.Interface.layout ℝ .bf16 p).inputs

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (p.head) → ℝ

def outputsEquiv (p : Parameters) (work : List Program) :
    (Interface.realIO p work).Outputs ≃ Outputs p where
  toFun := fun values b h row col => values (0 : Fin 1) (b,h,row,col,())
  invFun := fun values => Fin.cases
    (fun j => values j.1 j.2.1 j.2.2.1 j.2.2.2.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro values
    funext o j
    have zero : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,h,row,col,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl


def read (xs : Backing p) := VeriTile.Bench.Repairs.AttentionKernelAligned.Interface.readInputs .bf16 p xs.storage

def Backing.query (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.q (b*p.qBatchStride+h*p.qHeadStride+m*p.qRowStride+e*p.qColStride)

def Backing.key (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.k (b*p.kBatchStride+h*p.kHeadStride+e*p.kDimStride+m*p.kSeqStride)

def Backing.value (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.v (b*p.vBatchStride+h*p.vHeadStride+m*p.vSeqStride+e*p.vDimStride)

def Backing.bias (xs : Backing p) (batch : Fin p.batches) (head : Fin p.heads) (m e : Nat) : ℝ :=
  let b := batch.val
  let h := head.val
  read xs p.bias (b*p.biasBatchStride+h*p.biasHeadStride+m*p.biasRowStride+e)

/-- Mathematical input arrays in logical coordinates. Total extensions outside
finite windows retain the prior specification's exact alias convention. -/
structure Arrays (p : Parameters) where
  query : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  key : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  value : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  bias : Fin p.batches → Fin p.heads → Nat → Nat → ℝ

def arraysOfBacking (p : Parameters) (xs : Backing p) : Arrays p where
  query := xs.query
  key := xs.key
  value := xs.value
  bias := xs.bias

/-- The logical arrays and complete original windows are tied by an equality.
This keeps aliases, padding and unused cells without treating them as new
independent mathematical inputs. -/
structure Inputs (p : Parameters) where
  arrays : Arrays p
  backing : Backing p
  connection : arrays = arraysOfBacking p backing

def Inputs.query (xs : Inputs p) := xs.arrays.query
def Inputs.key (xs : Inputs p) := xs.arrays.key
def Inputs.value (xs : Inputs p) := xs.arrays.value
def Inputs.bias (xs : Inputs p) := xs.arrays.bias

def inputsEquiv (p : Parameters) (work : List Program) :
    (Interface.realIO p work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO ℝ Real.R (Inputs p) (Outputs p) where
  core := Interface.realIO p work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.LogicalIO
