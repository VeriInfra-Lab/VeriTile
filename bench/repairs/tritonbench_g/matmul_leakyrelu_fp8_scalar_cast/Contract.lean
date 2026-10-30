import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Tensor
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Reference

/-! Actual FP32 Tensor bindings. Signed descriptor/storage bounds are ordinary
Torch ABI facts, not extra Python admission guards. No allocator, host-return,
compiler success, numerical result, or whole-kernel execution is assumed. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
open VeriTile.Triton
open MatmulLeakyrelu.KernelOnly
open MatmulLeakyreluFp8
noncomputable section

structure Request where
  A : Matrix
  B : Matrix
  activation : String
  /-- Task-local observations of the existing torch.is_conj API. -/
  aConj : Bool
  bConj : Bool

/-- Symbolic view roles retain actual object, pointer, stride and storage offset.
Both input roles are populated from one physical reader, so A/B may alias. -/
def inputDescriptor (role : Nat) (x : Matrix) (conj : Bool) : Tensor where
  object := x.tensor.object
  storage := role
  offset := x.tensor.storageOffset
  rows := x.rows
  columns := x.columns
  rowStride := x.strideRow
  columnStride := x.strideColumn
  dtype := .fp32
  device := x.tensor.metadata.device.index
  isNeg := x.tensor.metadata.isNeg
  isConj := conj
  allocationBytes := x.storageBytes
  allocationBase := BitVec.ofNat 64 x.tensor.storage.base
  pointer := x.tensor.dataPtr

def outputDescriptor (r : Request) (c : OutputTensor) : Tensor where
  object := c.object
  storage := 2
  offset := c.storageOffset
  rows := r.A.rows
  columns := r.B.columns
  rowStride := max r.B.columns 1
  columnStride := 1
  dtype := .fp16
  device := c.metadata.device.index
  isNeg := c.metadata.isNeg
  isConj := Bool.false
  allocationBytes := 2*c.storageElements
  allocationBase := BitVec.ofNat 64 c.storageBase
  pointer := c.metadata.dataPtr

def arguments (r : Request) : Host.Arguments :=
  ⟨inputDescriptor 0 r.A r.aConj, inputDescriptor 1 r.B r.bConj, r.activation⟩

structure Contract (r : Request) (c : OutputTensor) : Prop where
  aAPI : MatrixLaws r.A
  bAPI : MatrixLaws r.B
  outputAPI : OutputLaws c r.A r.B
  dimensions : r.A.columns = r.B.rows
  cuda : r.A.tensor.metadata.device.kind = "cuda"
  sameDevice : r.B.tensor.metadata.device = r.A.tensor.metadata.device
  contiguous : r.A.tensor.metadata.isContiguous = Bool.true
  aResolved : r.A.tensor.metadata.isNeg = Bool.false ∧ r.aConj = Bool.false
  bResolved : r.B.tensor.metadata.isNeg = Bool.false ∧ r.bConj = Bool.false
  aAlignment : r.A.tensor.metadata.numel ≠ 0 → r.A.tensor.dataPtr.toNat%4 = 0
  bAlignment : r.B.tensor.metadata.numel ≠ 0 → r.B.tensor.dataPtr.toNat%4 = 0
  aABI : (inputDescriptor 0 r.A r.aConj).Bounded
  bABI : (inputDescriptor 1 r.B r.bConj).Bounded
  outputABI : (outputDescriptor r c).Bounded
  fresh : Fresh r.A r.B c

/-- No input/input storage separation is required. The view-role separation
here follows from the chosen symbolic names; physical freshness is Contract.fresh. -/
def layout (r : Request) (c : OutputTensor) (h : Contract r c) :
    Layout (arguments r).problem ⟨(arguments r).A,(arguments r).B,outputDescriptor r c⟩ where
  aShape := ⟨rfl,rfl⟩
  bShape := ⟨h.dimensions.symm,rfl⟩
  cShape := ⟨rfl,rfl⟩
  aType := rfl
  bType := rfl
  cType := rfl
  aBound := h.aABI
  bBound := h.bABI
  cBound := h.outputABI
  outputStrides := ⟨rfl,rfl⟩
  aSeparate := by change (2 : Nat) ≠ 0; decide
  bSeparate := by change (2 : Nat) ≠ 1; decide

/-- Initial symbolic views use a common physical address reader. Thus aliases
agree definitionally and the initial state contains no desired-result values. -/
def initial {α : Type} (r : Request) (read : Nat → α) (scratch : Memory α) : Memory α :=
  fun point => if point.1 = 0 then read (r.A.tensor.storage.base+4*point.2)
    else if point.1 = 1 then read (r.B.tensor.storage.base+4*point.2)
    else scratch point

theorem initial_a {α : Type} (r : Request) (h : Contract r c)
    (read : Nat → α) (scratch : Memory α) (i k : Nat)
    (hi : i < r.A.rows) (hk : k < r.A.columns) :
    initial r read scratch ((arguments r).A.address i k) = logical read r.A i k := by
  simp [initial, arguments, inputDescriptor, Tensor.address, logical,
    h.aAPI.indices i hi k hk, VeriTile.Torch.FP32.Storage.address]

theorem initial_b {α : Type} (r : Request) (h : Contract r c)
    (read : Nat → α) (scratch : Memory α) (k j : Nat)
    (hk : k < r.A.columns) (hj : j < r.B.columns) :
    initial r read scratch ((arguments r).B.address k j) = logical read r.B k j := by
  simp [initial, arguments, inputDescriptor, Tensor.address, logical,
    h.bAPI.indices k (by simpa [←h.dimensions] using hk) j hj,
    VeriTile.Torch.FP32.Storage.address]

theorem output_coordinate (r : Request) (c : OutputTensor) (h : Contract r c)
    (i j : Nat) (hi : i < r.A.rows) (hj : j < r.B.columns) :
    (outputDescriptor r c).address i j = (2,c.storageOffset+i*r.B.columns+j) ∧
    outputAddress c (i*r.B.columns+j) = c.storageBase+2*((outputDescriptor r c).address i j).2 := by
  have hn : 1 ≤ r.B.columns := by omega
  have hb : i*r.B.columns+j < c.metadata.numel := by
    rw [output_numel r.A r.B c h.outputAPI]
    exact flat_bound _ _ _ _ hi hj
  constructor
  · simp [outputDescriptor, Tensor.address, max_eq_left hn]
  · simp [outputAddress, h.outputAPI.indices _ hb, outputDescriptor,
      Tensor.address, max_eq_left hn, Nat.add_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
