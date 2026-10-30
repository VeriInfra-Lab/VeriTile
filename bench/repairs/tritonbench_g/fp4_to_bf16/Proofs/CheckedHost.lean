import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Contract
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Host

/- The source launcher selects x.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.Checked
open VeriTile.Torch VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule
open Bytes Connection
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The same Tensor's observed shape, pointer (already including offset), and
device feed the historical host/source model. These are metadata identities,
not additional input restrictions or assumptions about conversion results. -/
structure MetadataBinding (x : TensorMetadata) (h : Input)
    (s : Finite.Storage h.shape.count) (device : Nat) : Prop where
  shape : x.shape = h.shape.input
  pointer : x.dataPtr.toNat = s.inputPointer
  device : x.device.index = device

theorem numel_binding (x : TensorMetadata) (h : Input)
    (s : Finite.Storage h.shape.count) (device : Nat) (binding : MetadataBinding x h s device) :
    x.numel = h.shape.count := by
  rw [TensorMetadata.numel,binding.shape,input_numel]

/-- Ordinary Tensor storage semantics: a contiguous uint8 Tensor without a
lazy negative flag reads its logical bytes directly at data_ptr+i. No result,
kernel success, or FP4 conversion statement occurs in this relation. -/
structure LogicalStorage (x : TensorMetadata) (heap : Heap) (logical : Nat → BitVec 8) : Prop where
  read : x.dtype = .other "uint8" → x.isContiguous = Bool.true → x.isNeg = Bool.false →
    ∀ i,i<x.numel → heap (x.dataPtr.toNat+i) = logical i

def logicalNibble (logical : Nat → BitVec 8) (j : Nat) : Bits.Nibble :=
  if j%2=0 then Bits.high (logical (j/2)) else Bits.low (logical (j/2))

theorem logical_nibble (x : TensorMetadata) (h : Input) (s : Finite.Storage h.shape.count)
    (device : Nat) (binding : MetadataBinding x h s device)
    (heap : Heap) (logical : Nat → BitVec 8) (storage : LogicalStorage x heap logical)
    (checked : checkInputs (.tensor x) = .ok ()) (j : Nat) (hj : j<2*x.numel) :
    targetNibble s heap j = logicalNibble logical j := by
  have contract : InputContract x := (checkInputs_iff (.tensor x)).mp checked
  have raw := storage.read contract.dtype contract.contiguous contract.noNeg (j/2) (by omega)
  rw [binding.pointer] at raw
  simp only [targetNibble,logicalNibble,raw]

inductive CheckedFailure where
  | input (reason : InputError)
  | runtime (reason : Failure)

/-- Checks precede the accepted allocation/assert/empty/launch protocol.
Fresh output CUDA and allocation/setup outcomes are ordinary API facts. -/
def runChecked (rest : Algebra Word) (x : ArgumentMetadata) (h : Input)
    (s : Finite.Storage h.shape.count) (heap : Heap) (initial : State Word)
    (identity device : Nat) (jobs : List Nat) (allocationReady outputCuda setupReady : Bool) :
    Option (Except CheckedFailure Observation) :=
  match checkInputs x with
  | .error e => some (.error (.input e))
  | .ok () =>
    match x with
    | .nonTensor => none
    | .tensor x =>
      (observe rest h s heap initial identity device jobs allocationReady x.isContiguous
        x.isCuda outputCuda setupReady).map (Except.mapError CheckedFailure.runtime)

theorem checked_source (rest : Algebra Word) (x : TensorMetadata)
    (checked : checkInputs (.tensor x) = .ok ()) (h : Input)
    (s : Finite.Storage h.shape.count) (heap : Heap) (initial : State Word)
    (identity device : Nat) (jobs : List Nat) (allocationReady outputCuda setupReady : Bool) :
    runChecked rest (.tensor x) h s heap initial identity device jobs allocationReady outputCuda setupReady =
      (observe rest h s heap initial identity device jobs allocationReady Bool.true Bool.true outputCuda setupReady).map
        (Except.mapError CheckedFailure.runtime) := by
  have contract : InputContract x := (checkInputs_iff (.tensor x)).mp checked
  simp [runChecked,checked,contract.contiguous,TensorMetadata.isCuda,contract.cuda]

theorem rejected (rest : Algebra Word) (x : ArgumentMetadata) (e : InputError)
    (bad : checkInputs x = .error e) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (identity device : Nat) (jobs : List Nat)
    (allocationReady outputCuda setupReady : Bool) :
    runChecked rest x h s heap initial identity device jobs allocationReady outputCuda setupReady =
      some (.error (.input e)) := by simp [runChecked,bad]

#axiomsClean checked_source
#axiomsClean rejected
end
end VeriTile.Bench.Repairs.Fp4ToBf16.Checked
