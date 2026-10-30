import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.attn_fwd_causal.Proofs.AcceptedReal
import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeReference
import bench.tritonbench_g.attn_fwd_causal.Proofs.Physical
import VeriTile.Triton.Float.ObservedProgramIO

/-! Complete repaired attention source and every legal grid order. The source
fixes FP32 Q/K/V/scales, TF32 QK, FP16 P/V and PV, FP32 carry, and FP16 output.
Input cell channels use Structural's erased `.real` FP32 read type. Read-only
input aliases are permitted; only output/input separation is required. -/
namespace VeriTile.Bench.Repairs.AttnFwdCausal.IO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

def inputs (p : Parameters) : List IOChannel :=
  [⟨.real,[p.batches,p.heads,p.length,128],p.Q⟩,
   ⟨.real,[p.batches,p.heads,p.length,128],p.K⟩,
   ⟨.real,[p.batches,p.heads,p.length,128],p.V⟩,
   ⟨.real,[p.batches*p.heads,(p.length+127)/128],p.QS⟩,
   ⟨.real,[p.batches*p.heads,(p.length+63)/64],p.KS⟩]

abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputs p)

def inputAddress (p : Parameters) (i : Fin (inputs p).length) :
    TileIndex ((inputs p)[i]).shape → Nat :=
  Fin.cases (fun j => j.1.val*p.q.batch+j.2.1.val*p.q.head+j.2.2.1.val*p.q.row+j.2.2.2.1.val*p.q.dim)
    (Fin.cases (fun j => j.1.val*p.k.batch+j.2.1.val*p.k.head+j.2.2.2.1.val*p.k.dim+j.2.2.1.val*p.k.row)
      (Fin.cases (fun j => j.1.val*p.v.batch+j.2.1.val*p.v.head+j.2.2.1.val*p.v.row+j.2.2.2.1.val*p.v.dim)
        (Fin.cases (fun j => j.1.val*((p.length+127)/128)+j.2.1.val)
          (Fin.cases (fun j => j.1.val*((p.length+63)/64)+j.2.1.val) (fun i => Fin.elim0 i))))) i

theorem input_dtype (p : Parameters) (i : Fin (inputs p).length) :
    (inputs p)[i].dtype=.real := by fin_cases i <;> rfl

def inputValue (p : Parameters) (vals : Inputs α p) (i : Fin (inputs p).length)
    (j : TileIndex ((inputs p)[i]).shape) : α :=
  cast (congrArg (Value α) (input_dtype p i)) (vals i j)

def launch (M : Algebra β) (dot : Dot β) (p : Parameters) (pid : Program) : Invocation β where
  kernel := kernel p
  backend := FP.ProfiledRewrite.engine (withDot M dot)
  clearRegisters := Bool.true
  pids := ids pid
  numPids := some (dimensions (grid p))

def program (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : Fin 128 → β → α) : ProgramIO α β where
  inputs := inputs p
  outputs := [⟨.fp16,[p.batches,p.heads,p.length,128],p.Out⟩]
  scratch := []
  buffers := [p.Q,p.K,p.V,p.QS,p.KS,p.Out]
  read := fun i _ => inputAddress p i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
    (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => work.map (launch M dot p)
  encode := encode
  observe := fun o => Fin.cases (fun j => observe j.2.2.2.1) (fun i => Fin.elim0 i) o

end
end VeriTile.Bench.Repairs.AttnFwdCausal.IO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.PairConnection
open VeriTile Triton FP.Structural Source Whole
noncomputable section

def kernel (p : Parameters) : ComputeKernel :=
  TypedSource.quotientRemainder p.Q p.K p.V p.QS p.KS p.Out
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.batches p.heads p.length 128 128 64 128 128 3

end
end VeriTile.Bench.Repairs.AttnFwdCausal.PairConnection

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Reference
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Finite channel lookup. Simultaneous IO bindings prove that overlapping
read-only ports give the same value. The default only extends unused addresses. -/
def HasValue (p : Parameters) (vals : IO.Inputs α p) (r : RegionName) (off : Nat) (x : α) : Prop :=
  ∃ i j, (IO.inputs p)[i].buf=r ∧ IO.inputAddress p i j=off ∧ IO.inputValue p vals i j=x

def inputAt [Inhabited α] (p : Parameters) (vals : IO.Inputs α p) (r : RegionName) (off : Nat) : α := by
  classical
  exact if h : ∃ x, HasValue p vals r off x then Classical.choose h else default

end
end VeriTile.Bench.Repairs.AttnFwdCausal.Reference

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RealIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- The complete source schedule in mathematical Real semantics. -/
def program (p : Parameters) (work : List Program) : ProgramIO ℝ Real.R :=
  IO.program Real.algebra Real.dotModel p work some (fun _ x=>x.unbotD 0)

end
end VeriTile.Bench.Repairs.AttnFwdCausal.RealIO

/-! Logical array coordinates for the existing complete IO. The storage field
retains every original input cell, including aliases and padding; the view is
an equivalence and never replaces the underlying source, schedule or frame. -/
namespace VeriTile.Bench.Repairs.AttnFwdCausal.LogicalIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

structure Backing (p : Parameters) where
  storage : IO.Inputs ℝ p

abbrev Outputs (p : Parameters) :=
  Fin p.batches → Fin p.heads → Fin p.length → Fin (128) → ℝ

def outputsEquiv (p : Parameters) (work : List Program) :
    (RealIO.program p work).Outputs ≃ Outputs p where
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


def read (xs : Backing p) := Reference.inputAt p xs.storage

def Backing.query (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.Q (b.val*p.q.batch+h.val*p.q.head+m*p.q.row+e*p.q.dim)

def Backing.key (xs : Backing p) (b : Fin p.batches) (h : Fin p.heads) (m e : Nat) : ℝ :=
  read xs p.K (b.val*p.k.batch+h.val*p.k.head+e*p.k.dim+m*p.k.row)

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
    (RealIO.program p work).Inputs ≃ Inputs p where
  toFun := fun values => ⟨arraysOfBacking p ⟨values⟩,⟨values⟩,rfl⟩
  invFun := fun xs => xs.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

def io (p : Parameters) (work : List Program) :
    LogicalProgramIO ℝ Real.R (Inputs p) (Outputs p) where
  core := RealIO.program p work
  inputs := inputsEquiv p work
  outputs := outputsEquiv p work

end
end VeriTile.Bench.Repairs.AttnFwdCausal.LogicalIO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Reference
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Independent mathematical causal softmax-weighted sum on the finite input
ports, with their admitted read-only aliases. It contains no source recurrence. -/
def mathematical (p : Parameters) (vals : IO.Inputs ℝ p)
    (j : TileIndex [p.batches,p.heads,p.length,128]) : ℝ :=
  Real.expected p (inputAt p vals) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val

def nativeInputs (p : Parameters) (vals : IO.Inputs Native.Word p) (batch head : Nat) : NativeReference.Inputs :=
  ⟨fun n d=>inputAt p vals p.Q (batch*p.q.batch+head*p.q.head+n*p.q.row+d*p.q.dim),
   fun n d=>inputAt p vals p.K (batch*p.k.batch+head*p.k.head+d*p.k.dim+n*p.k.row),
   fun n d=>inputAt p vals p.V (batch*p.v.batch+head*p.v.head+n*p.v.row+d*p.v.dim),
   fun block=>inputAt p vals p.QS ((batch*p.heads+head)*((p.length+127)/128)+block),
   fun block=>inputAt p vals p.KS ((batch*p.heads+head)*((p.length+63)/64)+block)⟩

#specNonCircular mathematical avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Reference
