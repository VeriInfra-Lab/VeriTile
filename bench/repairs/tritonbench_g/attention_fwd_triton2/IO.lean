import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Raw
import bench.repairs.tritonbench_g.attention_fwd_triton2.Kernels
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Explicit full-grid IO. Five FP32 input storage channels retain legal read
aliases; BF16 outputs use the original four-dimensional strides. The source
itself keeps TF32 QK, half P/V/PV, FP32 online state and BF16 conversion. Each
actual launch clears local registers and installs the original 3D metadata. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Interface
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
open Classical

abbrev inputChannels (p : Parameters) : List IOChannel :=
  [⟨.real,[words p],p.Q⟩,⟨.real,[words p],p.K⟩,⟨.real,[words p],p.V⟩,
   ⟨.real,[p.batches*p.heads*((p.length+127)/128)],p.QS⟩,
   ⟨.real,[p.batches*p.heads*((p.length+63)/64)],p.KS⟩]
abbrev outputChannels (p : Parameters) : List IOChannel :=
  [⟨.bf16,[p.batches,p.heads,p.length,96],p.Out⟩]

def invocation {β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters) (program : Program) : Invocation β where
  kernel := Source.kernel p
  backend := FP.ProfiledRewrite.engine (withDot M dot)
  clearRegisters := Bool.true
  pids := ids program
  numPids := some (dimensions (grid p))

def io {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters)
    (encode : α → β)
    (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α)
    (work : List Program) : ProgramIO α β where
  inputs := inputChannels p
  outputs := outputChannels p
  scratch := []
  buffers := [p.Q,p.K,p.V,p.QS,p.KS,p.Out]
  read := fun i _ j => by
    rcases i with ⟨i,hi⟩
    have bound : i<5 := hi
    interval_cases i <;> exact j.1.val
  inputMask := fun _ _ _ => True
  write := fun o _ j => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact offset p.o j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => work.map (invocation M dot p)
  encode := encode
  observe := fun o j => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact observe j

abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputChannels p)

/-- A deterministic finite logical reader. Overlapping input roles are pinned
to the same source cell by shared InputBinding; no read-alias exclusion is added. -/
def reader {α : Type} [Inhabited α] (p : Parameters) (vals : Inputs α p)
    (r : RegionName) (off : Nat) : α :=
  if h : r=p.Q ∧ off<words p then vals 0 (⟨off,h.2⟩,())
  else if h : r=p.K ∧ off<words p then vals 1 (⟨off,h.2⟩,())
  else if h : r=p.V ∧ off<words p then vals 2 (⟨off,h.2⟩,())
  else if h : r=p.QS ∧ off<p.batches*p.heads*((p.length+127)/128) then vals 3 (⟨off,h.2⟩,())
  else if h : r=p.KS ∧ off<p.batches*p.heads*((p.length+63)/64) then vals 4 (⟨off,h.2⟩,())
  else default

/-- Source-free exact-Real normalized base-2 attention formula. -/
def realFormula (p : Parameters) (vals : Inputs ℝ p) : ChannelValues ℝ (outputChannels p) :=
  fun o j => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact RealReturned.target p (reader p vals) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val

def nativeInput (word : BitVec 32) : Native.Expr := fun _ => .ready (.bits word)
def observe32 (I : KernelOnly.Instructions) (d : Nat) (value : Native.Expr) : BitVec 32 :=
  (Native.lower (value (Native.physicalOwner d))).eval I.isa

/-- Original physical-owner observation of the independent native recurrence;
only the final BF16 word is returned, with all primitive instruction ports explicit. -/
def nativeFormula (I : KernelOnly.Instructions) (p : Parameters) (vals : Inputs (BitVec 32) p) :
    ChannelValues (BitVec 16) (outputChannels p) :=
  fun o j => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact (observe32 I j.2.2.2.1.val
      (KernelOnly.Specification.algorithm Native.algebra Native.dotModel p
        (fun r off=>nativeInput (reader p vals r off))
        (j.2.2.1.val/128) j.1.val j.2.1.val (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt))).setWidth 16

end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Interface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Pair
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

def kernel (p : Parameters) : ComputeKernel :=
  CandidateSource.shifted p.Q p.K p.V p.QS p.KS p.Out
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.batches p.heads p.length 96 128 64 128 96 1

def invocation {α : Type} (M : Algebra α) (dot : Dot α) (p : Parameters) (program : Program) : Invocation α :=
  { Interface.invocation M dot p program with kernel := kernel p }

def variant {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters) (encode : α → β)
    (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α) (work : List Program) :
    ProgramIO.Variant (io M dot p encode observe work) where
  schedule := fun _ => work.map (invocation M dot p)
  observe := (io M dot p encode observe work).observe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Pair

/-! Logical array coordinates for the existing complete IO. The storage field
retains every original input cell, including aliases and padding; the view is
an equivalence and never replaces the underlying source, schedule or frame. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.LogicalIO
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

structure Backing (p : Parameters) where
  storage : Interface.Inputs ℝ p

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (96) → ℝ

def outputsEquiv (p : Parameters) (work : List Program) :
    (Interface.io Real.algebra Real.dotModel p some (fun _ v => v.unbotD 0) work).Outputs ≃ Outputs p where
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


def read (xs : Backing p) := Interface.reader p xs.storage

def Backing.query (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.Q (b.val*p.q.batch+h.val*p.q.head+m*p.q.row+e*p.q.dim)

def Backing.key (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.K (b.val*p.k.batch+h.val*p.k.head+m*p.k.row+e*p.k.dim)

def Backing.value (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.V (b.val*p.v.batch+h.val*p.v.head+m*p.v.row+e*p.v.dim)

def Backing.queryScale (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (block : Nat) : ℝ :=
  read xs p.QS ((b.val*p.heads+h.val)*((p.length+127)/128)+block)

def Backing.keyScale (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (block : Nat) : ℝ :=
  read xs p.KS ((b.val*p.heads+h.val)*((p.length+63)/64)+block)

/-- Mathematical input arrays in logical coordinates. Total extensions outside
finite windows retain the prior specification's exact alias convention. -/
structure Arrays (p : Parameters) where
  query : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  key : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  value : Fin p.batches → Fin p.heads → Nat → Nat → ℝ
  queryScale : Fin p.batches → Fin p.heads → Nat → ℝ
  keyScale : Fin p.batches → Fin p.heads → Nat → ℝ

def arraysOfBacking (p : Parameters) (xs : Backing p) : Arrays p where
  query := xs.query
  key := xs.key
  value := xs.value
  queryScale := xs.queryScale
  keyScale := xs.keyScale

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
def Inputs.queryScale (xs : Inputs p) := xs.arrays.queryScale
def Inputs.keyScale (xs : Inputs p) := xs.arrays.keyScale

def inputsEquiv (p : Parameters) (work : List Program) :
    (Interface.io Real.algebra Real.dotModel p some (fun _ v => v.unbotD 0) work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO ℝ Real.R (Inputs p) (Outputs p) where
  core := Interface.io Real.algebra Real.dotModel p some (fun _ v => v.unbotD 0) work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.LogicalIO
