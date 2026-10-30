import VeriTile.Torch.Tensor
import bench.tritonbench_g.masked_select.Proofs.NativeDomain

/-! Raw argument admission and the separate post-materialization alignment
check. Resolution/contiguity keep logical values; their ordinary API contracts
supply actual prepared storage. No raw-input stride or negative-bit restriction
is imposed on the FP32 payload. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MaskedSelect.Whole (broadcastShape)

inductive InputError | tensors | layout | dtype | cuda | device | negativeMask | shape | extent | alignment
  deriving DecidableEq, Repr

structure InputContract (x m : TensorMetadata) (shape : List Nat) : Prop where
  layout : x.layout = .strided ∧ m.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ m.dtype = .other "bool"
  cuda : x.device.kind = "cuda" ∧ m.device.kind = "cuda"
  device : x.device = m.device
  maskResolved : m.isNeg = Bool.false
  broadcast : broadcastShape x.shape m.shape = some shape
  extent : 0 < shape.prod ∧ shape.prod < 2^31

def checkTensorInputs (x m : TensorMetadata) : Except InputError (List Nat) :=
  if x.layout != .strided || m.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || m.dtype != .other "bool" then .error .dtype
  else if !(x.isCuda && m.isCuda) then .error .cuda
  else if x.device != m.device then .error .device
  else if m.isNeg then .error .negativeMask
  else match broadcastShape x.shape m.shape with
    | none => .error .shape
    | some shape => if shape.prod = 0 || shape.prod >= 2^31 then .error .extent
        else .ok shape

def checkInputs (x m : ArgumentMetadata) : Except InputError (List Nat) :=
  match x, m with
  | .tensor x, .tensor m => checkTensorInputs x m
  | _, _ => .error .tensors

def AdmittedArguments (x m : ArgumentMetadata) (shape : List Nat) : Prop :=
  match x, m with
  | .tensor x, .tensor m => InputContract x m shape
  | _, _ => False

/-- Actual data_ptr after broadcast, resolve_neg and contiguous. -/
def PreparedContract (inputPointer : Nat) : Prop := inputPointer % 4 = 0

def checkPrepared (inputPointer : Nat) : Except InputError Unit :=
  if inputPointer % 4 != 0 then .error .alignment else .ok ()

/-- The kernel consumes the already prepared contiguous FP32 payload, Bool mask
and inclusive Int64 prefix. This realization constrains only input cells, never
output values. The Nat prefix is its nonnegative mathematical projection;
`prefix64_exact` and `destination64_selected` prove native Int64 correspondence. -/
abbrev KernelInputs {A : Type} := @VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.Reads A

/-- Output is separate from the three read buffers. This records the actual
fresh output allocation contract, without proving an allocator implementation. -/
abbrev KernelSeparate := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.Separate

/-- Physical contiguous descriptors retain 4/1/8/4-byte storage widths, bounds,
alignment and fresh output. `KernelInputs` supplies their logical cell values. -/
abbrev KernelBuffers := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.NativeBuffers

end VeriTile.Bench.Repairs.MaskedSelect
