import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelModels
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Contract

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelTensor
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev TensorLaws := VeriTile.Torch.FP32.TensorLaws

/-- The necessary INT8 input uses the shared metadata API and ordinary byte
storage observations. FP32 scale/output tensors use the shared FP32 API below.
No framework allocator or storage implementation is introduced. -/
structure Int8Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storage : Nat
  base : Nat
  elements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure Int8Laws (t : Int8Tensor) : Prop where
  dtype : t.metadata.dtype = .other "int8"
  layout : t.metadata.layout = .strided
  span : t.base + t.elements ≤ 2^64
  bounds : ∀ i, i < t.metadata.numel → t.physicalIndex i < t.elements
  pointer : 0 < t.metadata.numel →
    t.metadata.dataPtr = BitVec.ofNat 64 (t.base + t.storageOffset)
  contiguous : t.metadata.isContiguous = Bool.true →
    ∀ i, i < t.metadata.numel → t.physicalIndex i = t.storageOffset + i

def int8Info (t : Int8Tensor) : Direct.TensorInfo := ⟨t.metadata, t.storage⟩
def fp32Info (t : Tensor) : Direct.TensorInfo := ⟨t.metadata, t.storage.object⟩

def parameters (a : Direct.TensorInfo) (b : Int8Tensor) (config : LaunchConfig) : Parameters :=
  ⟨"b", "scale", "fpb", Direct.columns a, Direct.columns (int8Info b), config.blockK, config.blockN⟩

theorem separate (a : Direct.TensorInfo) (b : Int8Tensor) (config : LaunchConfig) :
    KernelGrid.Separate (parameters a b config) := by
  simp [KernelGrid.Separate, parameters]

/-- Logical values of the checked resolved INT8 tensor. Its byte decoder
retains every signed INT8 word, including -128. -/
def int8Value (read : Nat → BitVec 8) (b : Int8Tensor) (i : Nat) : BitVec 8 :=
  read (b.base + b.physicalIndex i)

def scaleValue (read : Nat → Word) (scale : Tensor) (laws : TensorLaws scale) (i : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue read scale laws i

theorem b_numel (a : Direct.TensorInfo) (b : Int8Tensor) (scale : Tensor) (out : Option Direct.TensorInfo)
    (h : Direct.InputContract a (int8Info b) (fp32Info scale) out) :
    b.metadata.numel = Direct.columns a * Direct.columns (int8Info b) := by
  have hn := Direct.numel_matrix (int8Info b) h.rank.2.1
  change b.metadata.numel = Direct.rows (int8Info b) * Direct.columns (int8Info b) at hn
  rw [← h.shape.1] at hn
  exact hn

theorem scale_numel (a : Direct.TensorInfo) (b : Int8Tensor) (scale : Tensor) (out : Option Direct.TensorInfo)
    (h : Direct.InputContract a (int8Info b) (fp32Info scale) out) :
    scale.metadata.numel = Direct.columns (int8Info b) := by
  have hs := h.shape.2
  change scale.metadata.shape = [Direct.columns (int8Info b)] at hs
  simp [VeriTile.Torch.TensorMetadata.numel, hs]

/-- Ordinary initial byte/typed-memory decoding and scratch allocation API
bindings. Allocation freshness is a local API boundary, not a caller guard
or an assumed result. The actual source is the only producer of final cells. -/
structure Binding (a : Direct.TensorInfo) (b : Int8Tensor) (scale output : Tensor)
    (read8 : Nat → BitVec 8) (read32 : Nat → Word) (initial : State Word) : Prop where
  outputShape : output.metadata.shape = [Direct.columns a, Direct.columns (int8Info b)]
  outputContiguous : output.metadata.isContiguous = Bool.true
  outputNoNeg : output.metadata.isNeg = Bool.false
  outputDevice : output.metadata.device = scale.metadata.device
  freshA : output.storage.object ≠ a.storage
  freshB : output.storage.object ≠ b.storage
  freshScale : output.storage.object ≠ scale.storage.object
  outputAligned : 0 < output.metadata.numel → output.dataPtr.toNat % 4 = 0
  input : ∀ i, i < b.metadata.numel →
    initial.mem "b" i = .mk .int (read8 (b.base + b.storageOffset + i)).toInt
  scaleMemory : ∀ i, i < scale.metadata.numel →
    initial.mem "scale" i = .mk .real (read32 (scale.storage.address (scale.storageOffset + i)))
  bytes : ∀ i, i < scale.metadata.numel → ∀ j : Fin 4,
    read8 (scale.storage.address (scale.physicalIndex i) + j.val) =
      BitVec.ofNat 8 ((read32 (scale.storage.address (scale.physicalIndex i))).raw / 2^(8*j.val))
  awayB : ∀ i, i < output.metadata.numel → ∀ j, j < b.metadata.numel → ∀ k : Fin 4,
    output.storage.address (output.physicalIndex i) + k.val ≠ b.base + b.physicalIndex j
  awayScale : ∀ i, i < output.metadata.numel → ∀ j, j < scale.metadata.numel → ∀ k l : Fin 4,
    output.storage.address (output.physicalIndex i) + k.val ≠ scale.storage.address (scale.physicalIndex j) + l.val

theorem reads (a : Direct.TensorInfo) (b : Int8Tensor) (scale output : Tensor)
    (out : Option Direct.TensorInfo) (h : Direct.InputContract a (int8Info b) (fp32Info scale) out)
    (lb : Int8Laws b) (ls : TensorLaws scale) (config : LaunchConfig)
    (read8 : Nat → BitVec 8) (read32 : Nat → Word) (initial : State Word)
    (binding : Binding a b scale output read8 read32 initial) :
    Reads (parameters a b config) (int8Value read8 b) (scaleValue read32 scale ls) initial := by
  constructor
  · intro i hi
    have bound : i < b.metadata.numel := by rw [b_numel a b scale out h]; exact hi
    dsimp only [parameters]
    rw [binding.input i bound]
    simp [int8Value, lb.contiguous h.contiguous.2.1 i bound, Nat.add_assoc]
  · intro i hi
    have bound : i < scale.metadata.numel := by rw [scale_numel a b scale out h]; exact hi
    dsimp only [parameters]
    rw [binding.scaleMemory i bound]
    have resolved : scale.metadata.isNeg = Bool.false := h.noNeg.2.2
    simp [scaleValue, VeriTile.Torch.FP32.logicalValue, resolved,
      ls.contiguous h.contiguous.2.2 i bound]

theorem input_pointer (a : Direct.TensorInfo) (b : Int8Tensor) (scale : Tensor)
    (out : Option Direct.TensorInfo) (h : Direct.InputContract a (int8Info b) (fp32Info scale) out)
    (laws : Int8Laws b) (i : Nat) (hi : i < b.metadata.numel) :
    (b.metadata.dataPtr + BitVec.ofNat 64 i).toNat = b.base + b.physicalIndex i := by
  have pos : 0 < b.metadata.numel := by omega
  have ib := laws.bounds i hi
  have span := laws.span
  have ci := laws.contiguous h.contiguous.2.1 i hi
  rw [laws.pointer pos, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat, ci]
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem fp32_pointer (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    (t.dataPtr + BitVec.ofNat 64 (4*i)).toNat = t.storage.address (t.physicalIndex i) := by
  have pos : 0 < t.metadata.numel := by omega
  have ib := laws.inBounds i hi
  have span := t.storage.spanFits
  have ci := laws.contiguous hc i hi
  rw [laws.pointer pos, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat, ci]
  simp only [VeriTile.Torch.FP32.Storage.address] at *
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem output_numel (a : Direct.TensorInfo) (b : Int8Tensor) (scale output : Tensor)
    (read8 : Nat → BitVec 8) (read32 : Nat → Word) (initial : State Word)
    (binding : Binding a b scale output read8 read32 initial) :
    output.metadata.numel = Direct.columns a * Direct.columns (int8Info b) := by
  simp [VeriTile.Torch.TensorMetadata.numel, binding.outputShape]

/-- Physical output observations use the actual shared Tensor storage/index.
The typed FP32 store transports its complete word without another rounding. -/
def outputStore (output : Tensor) (final : State Word) (i : Nat) : Nat × Word :=
  (output.storage.address (output.physicalIndex i), (final.mem "fpb" i).read .real)

#axiomsClean reads
#axiomsClean input_pointer
#axiomsClean fp32_pointer
#axiomsClean output_numel
#specNonCircular int8Value avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular scaleValue avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelTensor
