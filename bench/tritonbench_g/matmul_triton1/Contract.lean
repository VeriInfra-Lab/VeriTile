import bench.tritonbench_g.matmul_triton1.Proofs.LogicalInputs
import bench.tritonbench_g.matmul_triton1.Kernels

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
noncomputable section

abbrev InputContract := Revision001.InputContract
abbrev Tensor := TensorBinding.Tensor
abbrev Laws := TensorBinding.Laws
abbrev Word := TensorBinding.Word
abbrev Memory (α : Type) := Nat → α

def base (t : Tensor) : Nat := t.storage.address t.storageOffset
def address (t : Tensor) (i : Nat) : Nat := base t + 4*i
def storageContains (t : Tensor) (q : Nat) : Prop :=
  t.storage.base ≤ q ∧ q < t.storage.base + 4*t.storage.elements

/-- Tensor metadata and spatial contracts for the actual selected launch.
Inputs may share storage. Successful grid size is a launch condition, not a
new Python admission guard. No output value or execution is assumed. -/
structure Launch where
  a : Tensor
  b : Tensor
  c : Tensor
  la : Laws a
  lb : Laws b
  lc : Laws c
  defaultFP32 : Bool
  admitted : InputContract defaultFP32 a.metadata b.metadata
  outputShape : c.metadata.shape = [rows a.metadata, columns b.metadata]
  outputContiguous : c.metadata.isContiguous = Bool.true
  outputResolved : c.metadata.isNeg = Bool.false
  outputDevice : c.metadata.device = a.metadata.device
  outputAligned : 0 < c.metadata.numel → c.dataPtr.toNat % 4 = 0
  ra : RegionName
  rb : RegionName
  rc : RegionName
  separateA : ra ≠ rc
  separateB : rb ≠ rc
  freshObjectA : c.object ≠ a.object
  freshObjectB : c.object ≠ b.object
  storageSeparate : ∀ t, t = a ∨ t = b → ∀ q,
    base c ≤ q → q < base c + 4*c.metadata.numel → ¬storageContains t q
  launch : ((rows a.metadata+32-1)/32)*((columns b.metadata+32-1)/32) < 2^31

def Launch.parameters (x : Launch) : Core.Parameters :=
  ⟨x.ra,x.rb,x.rc,rows x.a.metadata,columns x.b.metadata,columns x.a.metadata,
    columns x.a.metadata,1,columns x.b.metadata,1,columns x.b.metadata,1,32,32,32,1⟩

abbrev Index (x : Launch) := Fin (rows x.a.metadata) × Fin (columns x.b.metadata)
def offset (x : Launch) (i : Index x) : Nat := i.1.val*columns x.b.metadata+i.2.val

/-- Every source input/output region is initially backed by one reader. -/
structure Binding {α : Type} (x : Launch) (read : Memory Word) (decode : Word → α)
    (s : State α) : Prop where
  inputs : TensorBinding.Inputs x.a x.b x.la x.lb x.admitted.contiguous.1
    x.admitted.contiguous.2 x.ra x.rb read decode s
  output : ∀ i, i < x.c.metadata.numel →
    s.mem x.rc i = Cell.mk .real (decode (read (address x.c i)))

end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
