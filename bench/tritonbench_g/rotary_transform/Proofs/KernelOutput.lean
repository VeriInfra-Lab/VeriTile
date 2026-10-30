import bench.tritonbench_g.rotary_transform.Proofs.KernelEntry
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelOutput
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Triton FP.Structural
noncomputable section

/-- Observation agreement for actual Tensor handles. Storage span/strides and
object identity use the existing checked descriptor; these additional fields
connect the shared Tensor metadata API without any output-value premise. -/
structure Observes (t : Checked.Tensor) (handle : VeriTile.Torch.TensorMetadata) : Prop where
  pointer : handle.dataPtr.toNat=t.base
  shape : handle.shape=t.shape
  device : handle.device.index=t.device
  cuda : handle.device.kind="cuda"
  strided : handle.layout=.strided
  contiguous : handle.isContiguous=t.contiguous
  noNeg : handle.isNeg=t.negative
  dtype : handle.dtype=if t.dtype="float32" then .floating .fp32 else .other t.dtype

def descriptor (c : Checked.Call) (out : Binding.Output c) : RotaryTransform.Region → Checked.Tensor
  | .input => c.input | .output => out.tensor
  | .cosine => c.cosine | .sine => c.sine
  | .cumulative => c.cumulative | .offsets => c.offsets
  | .other 0 => c.cumulative | .other 1 => c.offsets
  | .other 2 => c.cosine | .other 3 => c.sine
  | .other _ => c.input

def handles (c : Checked.Call) : List RotaryTransform.Region :=
  [.input,.cosine,.sine]++(if c.host.shape.varlen then [.cumulative] else [])++
    (if c.host.shape.offsetTensor then [.offsets] else [])

structure Inputs (c : Checked.Call) (out : Binding.Output c)
    (tensor : RotaryTransform.Region→VeriTile.Torch.TensorMetadata) : Prop where
  observed : ∀ r,r∈handles c → Observes (descriptor c out r) (tensor r)

/-- Read through actual descriptor pointers. Real interpretation supplies
arbitrary real input values, with no encoding of Real values into FP32. -/
def inputMemory {M α β : Type} (api : Ops M α β) (c : Checked.Call)
    (out : Binding.Output c) (m : M) : Memory α :=
  fun r i => api.read m (Binding.key (descriptor c out r) i)

/-- Finite output stores depend solely on the completed quotation state.
The original suffix copy is already derived in KernelEntry.suffix. -/
def serialize {M α β γ : Type} [Inhabited α] (api : Ops M β γ)
    (c : Checked.Call) (out : Binding.Output c) (project : α→β) (final : State α) (m : M) : M :=
  stores api (Binding.key out.tensor)
    (fun i => project ((final.mem (KernelAlgebra.regionName c.host.shape.output) i).read .real))
    (List.range (Host.size c.host.shape)) m

theorem read_serialize {M α β γ : Type} [Inhabited α] (api : Ops M β γ)
    (c : Checked.Call) (out : Binding.Output c) (project : α→β) (final : State α) (m : M)
    (i : Nat) (inside : i<Host.size c.host.shape) :
    api.read (serialize api c out project final m) (Binding.key out.tensor i)=
      project ((final.mem (KernelAlgebra.regionName c.host.shape.output) i).read .real) := by
  apply readback api _ _ _ m i (List.mem_range.mpr inside)
  · intro j _ equal
    have same : j=i := by
      by_contra ne
      exact Key.disjoint_ne (Binding.indices_apart out.tensor j i ne) equal
    rw [same]
  · intro j _ different
    exact Binding.indices_apart out.tensor i j (by intro same; subst j; exact different rfl)

theorem serialize_frame {M α β γ : Type} [Inhabited α] (api : Ops M β γ)
    (c : Checked.Call) (out : Binding.Output c) (project : α→β) (final : State α) (m : M)
    (owner address : Nat) (outside : Binding.Outside c out owner address) :
    api.observe (serialize api c out project final m) owner address=api.observe m owner address := by
  apply bytes_frame
  intro i hi
  exact outside i (List.mem_range.mp hi)

/-- Local FP32 codec, applied only to initial raw-word input observations. -/
theorem native_input_read {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (codec : Binding.Codec api) (c : Checked.Call) (out : Binding.Output c) (m : M)
    (r : RotaryTransform.Region) (i : Nat) :
    Physical.load (Binding.mirror api c out m) r i=inputMemory api c out m r i := by
  rw [←Physical.decode_encode (inputMemory api c out m r i)]
  unfold Physical.load
  apply congrArg Physical.decode
  funext byte
  change api.observe m (descriptor c out r).device ((descriptor c out r).base+(4*i+byte.val))=
    Physical.encode (api.read m (Binding.key (descriptor c out r) i)) byte
  simpa [Binding.key,Nat.add_assoc] using codec.bytes m (Binding.key (descriptor c out r) i) byte

structure Result (M : Type) where
  tensor : VeriTile.Torch.TensorMetadata
  object : Nat
  deviceRestored : Nat
  memory : M

def nativeEntry {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (c : Checked.Call) (out : Binding.Output c) (m : M) (other : State Native.Node) : State Native.Node :=
  let bytes := Binding.mirror api c out m
  KernelEntry.lift c bytes
    (fun r i => Native.atom (Physical.callerWords c.host.shape c.views bytes r i)) other

def nativeInvoke {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (c : Checked.Call) (out : Binding.Output c) (tensor : VeriTile.Torch.TensorMetadata)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) (other : State Native.Node) : Option (Result M) :=
  let bytes := Binding.mirror api c out m
  if check c bytes && Binding.pointerCheck c out bytes then
    (TypedGrid.run (Native.ops p) (Checked.configuration c bytes)
      c.host.abi.cumulative64 c.host.abi.offsets64 (nativeEntry api c out m other)).map
      (fun final => ⟨tensor,out.tensor.object,callerDevice,serialize api c out Native.Node.value final m⟩)
  else none

def realEntry {M β : Type} (api : Ops M ℝ β) (c : Checked.Call) (out : Binding.Output c)
    (bytes : Physical.Bytes) (m : M) (other : State ℝ) : State ℝ :=
  KernelEntry.lift c bytes (RealWrapper.prepared c (inputMemory api c out m)) other

def realInvoke {M β : Type} (api : Ops M ℝ β) (c : Checked.Call) (out : Binding.Output c)
    (tensor : VeriTile.Torch.TensorMetadata) (callerDevice : Nat) (bytes : Physical.Bytes)
    (m : M) (other : State ℝ) : Option (Result M) :=
  if check c bytes && Binding.pointerCheck c out bytes then
    (TypedGrid.run realOps (Checked.configuration c bytes) c.host.abi.cumulative64
      c.host.abi.offsets64 (realEntry api c out bytes m other)).map
      (fun final => ⟨tensor,out.tensor.object,callerDevice,serialize api c out id final m⟩)
  else none

#axiomsClean read_serialize
#axiomsClean serialize_frame
#axiomsClean native_input_read
#specNonCircular serialize avoiding [realFormula,nativeFormula,source,Native.graph]
#specNonCircular inputMemory avoiding [TypedGrid.run,ProgramModel.result,realInvoke,nativeInvoke]
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.KernelOutput
