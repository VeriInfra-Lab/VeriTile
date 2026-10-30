import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.attention_kernel.Proofs.RawKernel
import bench.repairs.tritonbench_g.attention_kernel.Proofs.RealReturned
import VeriTile.Triton.Float.ProgramIO

/-! Four logical input tensors with the source's original strides. Aliases
between inputs remain allowed; shared input binding enforces consistency at
shared addresses. Output is the complete [B,H,N,D] tensor. Both source-selected
16-bit formats, the full caller-selected grid order and empty grids are kept. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.IO
open VeriTile Triton FP.Structural
open VeriTile.Triton.FP.ProfiledRewrite (engine)
open Whole001 Source Whole Machine Whole002
noncomputable section

abbrev TensorShape (p : Parameters) : TileShape := [p.batches,p.heads,p.length,p.dim]
abbrev BiasShape (p : Parameters) : TileShape := [p.batches,p.heads,p.length,128]
def inputs (p : Parameters) : List IOChannel :=
  [⟨p.storage.tile,TensorShape p,p.Q⟩,⟨p.storage.tile,TensorShape p,p.K⟩,
   ⟨p.storage.tile,TensorShape p,p.V⟩,⟨p.storage.tile,BiasShape p,p.Bias⟩]
def outputs (p : Parameters) : List IOChannel := [⟨p.storage.tile,TensorShape p,p.Out⟩]

def address (p : Parameters) (strides : Strides) (i : TileIndex (TensorShape p)) : Nat :=
  offset strides i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
def biasAddress (p : Parameters) (i : TileIndex (BiasShape p)) : Nat :=
  i.1.val*p.bias.batch+i.2.1.val*p.bias.head+i.2.2.1.val*p.bias.row+i.2.2.2.1.val

def job {α : Type} (M : Algebra α) (dot : Dot α) (p : Parameters) (pid : Program) : Invocation α where
  kernel:=Source.kernel p
  backend:=engine (withDot M dot)
  clearRegisters:=Bool.true
  pids:=ids pid
  numPids:=some (dimensions (grid p))

def attentionIO {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters)
    (work : List Program) (encode : α → β)
    (observe : TileIndex (TensorShape p) → β → α) : ProgramIO α β where
  scratch:=[]
  inputs:=inputs p
  outputs:=outputs p
  buffers:=[p.Q,p.K,p.V,p.Bias,p.Out]
  read:=by
    intro i vals j
    change Fin 4 at i
    by_cases h0 : i=0
    · subst i; exact address p p.q j
    by_cases h1 : i=1
    · subst i; exact address p p.k j
    by_cases h2 : i=2
    · subst i; exact address p p.v j
    have h3 : i=3 := by omega
    subst i; exact biasAddress p j
  inputMask:=fun _ _ _=>True
  write:=by
    intro o vals j
    change Fin 1 at o
    have hz : o=0 := Fin.eq_zero o
    subst o
    exact address p p.o j
  outputMask:=fun _ _ _=>True
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>work.map (job M dot p)
  encode:=encode
  observe:=by
    intro o j
    change Fin 1 at o
    have hz : o=0 := Fin.eq_zero o
    subst o
    exact observe j

def realIO (p : Parameters) (work : List Program) : ProgramIO ℝ Real.R :=
  attentionIO Real.algebra Real.dotModel p work some (fun _ x=>x.unbotD 0)

def encodeWord (w : BitVec 16) : FamilyNative.Expr := fun _=>.ready (.bits (w.zeroExtend 32))
def observeWord (I : KernelOnly.Instructions) (layout : FamilyNative.Layout) (p : Parameters)
    (i : TileIndex (TensorShape p)) (x : FamilyNative.Expr) : BitVec 16 :=
  ((Native.lower (x (FamilyNative.physicalOwner layout i.2.2.2.1.val))).eval I.isa).setWidth 16

def nativeIO (I : KernelOnly.Instructions) (layout : FamilyNative.Layout)
    (p : Parameters) (work : List Program) : ProgramIO (BitVec 16) FamilyNative.Expr :=
  attentionIO (FamilyNative.algebra layout) (FamilyNative.dotModel layout) p work
    encodeWord (observeWord I layout p)

/-- Decode a logical typed input; all channels retain the selected FP16/BF16 tag. -/
def queryValues {α : Type} (p : Parameters) (vals : ChannelValues α (inputs p)) :
    TileIndex (TensorShape p) → α := fun i=>toFloat p.storage.float (vals (0 : Fin 4) i)
def keyValues {α : Type} (p : Parameters) (vals : ChannelValues α (inputs p)) :
    TileIndex (TensorShape p) → α := fun i=>toFloat p.storage.float (vals (1 : Fin 4) i)
def valueValues {α : Type} (p : Parameters) (vals : ChannelValues α (inputs p)) :
    TileIndex (TensorShape p) → α := fun i=>toFloat p.storage.float (vals (2 : Fin 4) i)
def biasValues {α : Type} (p : Parameters) (vals : ChannelValues α (inputs p)) :
    TileIndex (BiasShape p) → α := fun i=>toFloat p.storage.float (vals (3 : Fin 4) i)
def result {α : Type} (p : Parameters) (f : TileIndex (TensorShape p) → α) :
    ChannelValues α (outputs p) := by
  intro o j
  change Fin 1 at o
  have hz : o=0 := Fin.eq_zero o
  subst o
  exact ofFloat p.storage.float (f j)

end
end VeriTile.Bench.Repairs.AttentionKernel.IO

namespace VeriTile.Bench.Repairs.AttentionKernel.Target
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section

structure Views (α : Type) where
  q : Nat → Nat → α
  k : Nat → Nat → α
  v : Nat → Nat → α
  bias : Nat → Nat → α

def get {α : Type} [Inhabited α] {m n : Nat}
    (a : Fin m → Fin n → α) (r c : Nat) : α :=
  if hr : r<m then if hc : c<n then a ⟨r,hr⟩ ⟨c,hc⟩ else default else default

def views {α : Type} [Inhabited α] (p : Parameters) (vals : ChannelValues α (IO.inputs p))
    (b : Fin p.batches) (h : Fin p.heads) : Views α where
  q:=get (fun r d=>IO.queryValues p vals (b,h,r,d,()))
  k:=get (fun r d=>IO.keyValues p vals (b,h,r,d,()))
  v:=get (fun r d=>IO.valueValues p vals (b,h,r,d,()))
  bias:=get (fun r c=>IO.biasValues p vals (b,h,r,c,()))

end
end VeriTile.Bench.Repairs.AttentionKernel.Target

/-! Logical array coordinates for the existing complete IO. The storage field
retains every original input cell, including aliases and padding; the view is
an equivalence and never replaces the underlying source, schedule or frame. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.LogicalIO
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section

structure Backing (p : Parameters) where
  storage : ChannelValues ℝ (IO.inputs p)

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (p.dim) → ℝ

def storageEquiv (storage : TypedSource.Storage) : Value ℝ storage.tile ≃ ℝ where
  toFun := toFloat storage.float
  invFun := ofFloat storage.float
  left_inv := by cases storage <;> exact fun _ => rfl
  right_inv := by cases storage <;> exact fun _ => rfl

def outputsEquiv (p : Parameters) (work : List Program) :
    (IO.realIO p work).Outputs ≃ Outputs p where
  toFun := fun values b h row col => storageEquiv p.storage (values (0 : Fin 1) (b,h,row,col,()))
  invFun := fun values => Fin.cases
    (fun j => (storageEquiv p.storage).symm (values j.1 j.2.1 j.2.2.1 j.2.2.2.1)) (fun i => Fin.elim0 i)
  left_inv := by
    intro values
    funext o j
    have zero : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,h,row,col,⟨⟩⟩
    exact (storageEquiv p.storage).left_inv _
  right_inv := by
    intro values
    funext b h row col
    exact (storageEquiv p.storage).right_inv _


def Backing.query (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) :=
  (Target.views p xs.storage b h).q
def Backing.key (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) :=
  (Target.views p xs.storage b h).k
def Backing.value (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) :=
  (Target.views p xs.storage b h).v
def Backing.bias (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) :=
  (Target.views p xs.storage b h).bias

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
    (IO.realIO p work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO ℝ Real.R (Inputs p) (Outputs p) where
  core := IO.realIO p work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

end
end VeriTile.Bench.Repairs.AttentionKernel.LogicalIO
