import VeriTile.Triton.Memory.KernelSpec.Masked
import bench.tritonbench_g.softmax_triton2.Proofs.GuardedReciprocal
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.softmax_triton2.Kernels
import bench.tritonbench_g.softmax_triton1.Proofs.NativeGraph
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.RowIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Common strided row windows and tail masks. Encoding and observation belong
to the separate scalar/native program interpretations below. -/
abbrev memory (α : Type) (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : MemoryIO α where
  inputs := [⟨.real,[B],inp⟩]
  outputs := [⟨.real,[B],out⟩]
  scratch := []
  buffers := [inp,out]
  read := by
    intro i
    have h : i=0 := Fin.eq_zero i
    subst i
    exact fun _ j => ib+p*ir+j.1.val*ic
  inputMask := by
    intro i
    have h : i=0 := Fin.eq_zero i
    subst i
    exact fun _ j => j.1.val<n
  write := by
    intro o
    have h : o=0 := Fin.eq_zero o
    subst o
    exact fun _ j => ob+p*orow+j.1.val*oc
  outputMask := by
    intro o
    have h : o=0 := Fin.eq_zero o
    subst o
    exact fun _ j => j.1.val<n
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.RowIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1
open NativeGraph
noncomputable section

/-- The actual selected source and launch register state; V1 is the production layout. -/
def launch (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) : Invocation Node where
  kernel := SoftmaxTriton2.softmax_kernel out inp ib ob ir ic orow oc cols B
  backend := Floating.backend (algebra inst 1)
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then p else 0
  numPids := some (fun ax => if ax=0 then rows else 1)

/-- FP32 source channels use the Structural `.real` tag. Input words are constant
across threads; each output is observed at its actual blocked-layout owner. -/
@[reducible] def io (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) : ProgramIO Word Node where
  toMemoryIO := RowIO.memory Word out inp ib ob ir ic orow oc cols B p
  schedule := fun _ => [launch inst out inp rows cols ib ob ir ic orow oc B p]
  encode := atom
  observe := by
    intro o
    have h : o=0 := Fin.eq_zero o
    subst o
    exact fun j node => node (owner 1 B j.1.val)

/-- Logical word array; only source-masked slots are read by the native graph. -/
def words (B : Nat) (xs : TileIndex [B] → Word) (i : Nat) : Word :=
  if h : i<B then xs (⟨i,h⟩,PUnit.unit) else 0

/-- Source-independent ordered native formula, including exact padding. -/
def reference (inst : Instructions) (n B : Nat) (xs : TileIndex [B] → Word)
    (j : TileIndex [B]) : Word := NativeGraph.row inst 1 n B (words B xs) j.1.val

def pack (out : RegionName) (B : Nat) (f : TileIndex [B] → Word) :
    ChannelValues Word [⟨.real,[B],out⟩] := by
  intro o
  have h : o=0 := Fin.eq_zero o
  subst o
  exact f


end
end VeriTile.Bench.Repairs.SoftmaxTriton2.NativeIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
noncomputable section

/-- One actual row invocation, retaining storage offsets and complete active
read/write windows. The wrapper passes canonical contiguous strides. -/
def softmaxIO (out inp : RegionName) (ib ob n B : Nat) : MaskedKernelIO₁ where
  kernel := SoftmaxTriton2.softmax_kernel out inp ib ob n 1 n 1 n B
  projection := by rfl
  inp := inp
  out := out
  B := B
  read := fun pid => ib + pid*n
  write := fun pid => ob + pid*n
  mask := fun _ i => i.val < n

end
end VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open VeriTile.Triton FP.Structural FP.Guarded
open scoped VeriTile.Spec
noncomputable section

/-- The original kernel's public typed input/output windows. All raw scalar
payloads are admitted; this is metadata, not a numerical input predicate. -/
structure Layout where
  out : RegionName
  inp : RegionName
  ib : Nat
  ob : Nat
  ir : Nat
  ic : Nat
  orow : Nat
  oc : Nat
  n : Nat
  B : Nat
  p : Nat

structure IOProgram where
  kernel : ComputeKernel
  layout : Layout

/-- One canonical parameter binding for the original strided row layout. -/
def rowLayout (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : Layout :=
  ⟨out,inp,ib,ob,ir,ic,orow,oc,n,B,p⟩

def sourceProgram (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : IOProgram where
  kernel := softmax_kernel out inp ib ob ir ic orow oc n B
  layout := rowLayout out inp ib ob ir ic orow oc n B p

def reciprocalProgram (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : IOProgram where
  kernel := guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B
  layout := rowLayout out inp ib ob ir ic orow oc n B p

def invocation {α : Type} (M : Algebra α) (program : IOProgram) : Invocation α where
  kernel := program.kernel
  backend := FP.ProfiledRewrite.engine M
  clearRegisters := Bool.true
  pids := fun ax => if ax = 0 then program.layout.p else 0

/-- Exactly the original active strided input/output cells. FP32 source
precision stays in the concrete kernel; `.real` is the opaque scalar cell tag. -/
@[reducible] def sourceIO {α : Type} (M : Algebra α) (program : IOProgram) : ProgramIO α α where
  toMemoryIO := RowIO.memory α program.layout.out program.layout.inp
    program.layout.ib program.layout.ob program.layout.ir program.layout.ic
    program.layout.orow program.layout.oc program.layout.n program.layout.B program.layout.p
  schedule := fun _ => [invocation M program]
  encode := id
  observe := fun _ _ => id

def sourceVariant {α : Type} (M : Algebra α) (lhs rhs : IOProgram) :
    ProgramIO.Variant (sourceIO M lhs) where
  schedule := fun _ => [invocation M rhs]
  observe := fun _ _ => id

/-- Full existing kernel IO scope: both actual traces are safe, both runs
succeed, active complete typed output cells agree and both exterior frames hold.
The independent logical input binding is retained even though the memory proof
is stronger and applies to every initial scalar payload. -/
def IOEquivalent (R : Spec.Assumptions FP.GuardedFragment) (lhs rhs : IOProgram) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    Ports M D → (sourceIO M lhs).Equivalent (sourceVariant M lhs rhs)

instance : Spec.ProgramSyntax IOProgram where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × Layout
  signature p := (match p.kernel with | .mk inputs outputs _ => (inputs,outputs),p.layout)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := IOEquivalent
  sameContext := Eq

end
end VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.Logical
open VeriTile.Triton
noncomputable section

/-- The active logical subrow. The public IO still retains all B tile values,
its original mask and the complete execution/safety/frame judgment. -/
def active {n B : Nat} (cover : n ≤ B) (xs : Fin B → ℝ) (j : Fin n) : ℝ :=
  xs (Fin.castLE cover j)

end
end VeriTile.Bench.Repairs.SoftmaxTriton2.Logical
