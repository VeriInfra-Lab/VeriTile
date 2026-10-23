import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelTensor
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelReal
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelSchedule

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelPhysical
open VeriTile Triton FP.Structural
namespace BM
export VeriTile.Bench.Repairs.StreamK.Checked.ByteMemory
  (View Read read32 byte render render_at render_read render_frame pack_bytes Outside Separate)
end BM
noncomputable section
set_option maxHeartbeats 8000000

def view (b : Physical.Buffer) : BM.View := ⟨4*b.base,b.words⟩
def kernelInputRegions : List RegionName := ["query","key","value","initial","incoming","terminalGradient"]

/-- Caller-provided buffers and ordinary API/storage facts. The historical
allocation descriptor is reused only for shapes and writable-memory resources;
allocation, dispatch, autograd and cotangent copying are outside this theorem. -/
structure Resources (c : Host.Invocation) (a : Host.Allocations) where
  api : Admission.TensorAPI c
  layout : Host.AllocationContract c a
  pointers : Admission.PointerObservation c a
  pointerChecks : Contract.PointerContract c a pointers
  separate : ∀ x∈Host.outputBuffers a,∀ y∈Host.outputBuffers a,x≠y → Physical.Separate x y
  inputFresh : ∀ x∈Host.inputBuffers c a,∀ y∈Host.outputBuffers a,Physical.Separate x y
  tensors : KernelTensor.Buffers c a
  inputStorage : ∀ r∈kernelInputRegions,∀ t,Host.descriptor c a r=some t →
    ∀ b∈Host.outputBuffers a,BM.Separate (view b)
      ⟨(tensors.actual t).storage.base,(tensors.actual t).storage.elements⟩

def Resources.physical (resources : Resources c a) : Host.PhysicalAllocation c a :=
  Admission.physicalAllocation c a resources.pointers
    ((Contract.pointers_iff c a resources.pointers).mpr resources.pointerChecks)
    resources.separate resources.inputFresh

/-- The executable guards imply finite launch axes, signed offset arithmetic
and actual pointer room. `Host.KernelAddresses` proves the ten source pointer
orientations satisfy `formedLimit`, including masked lanes and transposes.
Generic runtime grid-product arithmetic is outside this kernel contract. -/
structure AddressConformance (c : Host.Invocation) (a : Host.Allocations) : Prop where
  axes : ∀ stage,Host.AxisSafe (gridSize stage (c.parametersFor stage))
  formed : ∀ b∈Host.outputBuffers a++Host.inputBuffers c a,∀ o,o<Host.formedLimit c.dims →
    o<2^31 ∧ 4*(b.base+o)<2^63
  affine : ∀ base row column total,total=base+row+column → total<Host.formedLimit c.dims →
    base<2^31 ∧ row<2^31 ∧ column<2^31

theorem address_conformance (c : Host.Invocation) (flags : Admission.RuntimeFlags)
    (h : Contract.InputContract c flags) (a : Host.Allocations) (resources : Resources c a) :
    AddressConformance c a := by
  have admitted := Contract.admitted c flags h resources.api
  exact ⟨Host.launch_axes_safe c admitted,
    Host.actual_pointer_safe c admitted a resources.physical,
    Host.affine_component_safe c.dims admitted.machine⟩

def publishBytes (a : Host.Allocations) (values : RegionName → Nat → BitVec 32)
    (before : BM.Read) : BM.Read :=
  (Host.outputBuffers a).foldl (fun bytes b => BM.render bytes (view b) (fun i => ⟨values b.region i⟩)) before

theorem bytes_at (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (values : RegionName → Nat → BitVec 32) (before : BM.Read) (b : Physical.Buffer)
    (hb : b∈Host.outputBuffers a) (i j : Nat) (hi : i<b.words) (hj : j<4) :
    publishBytes a values before (4*b.base+4*i+j)=BM.byte (values b.region i) j := by
  apply VeriTile.Bench.Repairs.ChunkDeltaFwd.list_read
    _ (fun bytes : BM.Read => bytes (4*b.base+4*i+j)) (Host.outputBuffers a) b before _ hb
  · intro bytes
    exact BM.render_at bytes (view b) (fun i => ⟨values b.region i⟩) i j hi hj
  · intro other hother hn bytes
    apply BM.render_frame
    have separated := resources.separate b hb other hother (Ne.symm hn)
    unfold Physical.Separate at separated
    change 4*b.base+4*i+j<4*other.base ∨ 4*other.base+4*other.words≤4*b.base+4*i+j
    omega

theorem bytes_read (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (values : RegionName → Nat → BitVec 32) (before : BM.Read) (b : Physical.Buffer)
    (hb : b∈Host.outputBuffers a) (i : Nat) (hi : i<b.words) :
    BM.read32 (publishBytes a values before) (4*(b.base+i))=values b.region i := by
  rw [Nat.mul_add]
  unfold BM.read32
  have h0 := bytes_at c a resources values before b hb i 0 hi (by decide)
  simp only [Nat.add_zero] at h0
  rw [h0,bytes_at c a resources values before b hb i 1 hi (by decide),
    bytes_at c a resources values before b hb i 2 hi (by decide),
    bytes_at c a resources values before b hb i 3 hi (by decide)]
  exact BM.pack_bytes _

theorem bytes_frame (a : Host.Allocations) (values : RegionName → Nat → BitVec 32)
    (before : BM.Read) (address : Nat) (outside : ∀ b∈Host.outputBuffers a,BM.Outside (view b) address) :
    publishBytes a values before address=before address := by
  apply list_observe _ (fun bytes : BM.Read => bytes address)
  intro b hb bytes
  exact BM.render_frame bytes (view b) _ address (outside b hb)

theorem input_bytes (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (values : RegionName → Nat → BitVec 32) (before : BM.Read) (r : RegionName)
    (hr : r∈kernelInputRegions) (t : Host.Tensor) (ht : Host.descriptor c a r=some t)
    (address : Nat) (inside : (resources.tensors.actual t).storage.base≤address ∧
      address<(resources.tensors.actual t).storage.base+4*(resources.tensors.actual t).storage.elements) :
    publishBytes a values before address=before address := by
  apply bytes_frame
  intro b hb
  have separated := resources.inputStorage r hr t ht b hb
  unfold BM.Separate BM.Outside at *
  dsimp only at separated
  omega

theorem input_real (c : Host.Invocation) (a : Host.Allocations) (resources : Resources c a)
    (values : Memory ℝ) (before : Nat → Cell ℝ) (r : RegionName)
    (hr : r∈kernelInputRegions) (t : Host.Tensor) (ht : Host.descriptor c a r=some t)
    (address : Nat) (inside : (resources.tensors.actual t).storage.base≤4*address ∧
      4*address<(resources.tensors.actual t).storage.base+4*(resources.tensors.actual t).storage.elements) :
    Host.publish a values before address=before address := by
  apply Host.published_frame
  intro b hb
  have separated := resources.inputStorage r hr t ht b hb
  unfold BM.Separate at separated
  simp only [view] at separated
  unfold Physical.Contains
  omega

def runReal (c : Host.Invocation) (a : Host.Allocations) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    Option (Memory ℝ × (Nat → Cell ℝ)) := do
  let logical ← KernelExecution.run Real.algebra Real.dotPrimitive c.mathCall
    (KernelTensor.realMemory c a before fallback)
  return (logical,Host.publish a logical before)

def runNative (profile : NativeTerms.FP32Profile) (c : Host.Invocation) (a : Host.Allocations)
    (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) :
    Option ((RegionName → Nat → BitVec 32) × BM.Read) := do
  let logical ← KernelExecution.runNative profile c.mathCall c.scale.argumentFP32
    (KernelTensor.nativeMemory c a before fallback)
  return (logical,publishBytes a logical before)

theorem real_run (c : Host.Invocation) (a : Host.Allocations) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    runReal c a before fallback=some
      (c.mathCall.result (KernelTensor.realMemory c a before fallback),
       Host.publish a (c.mathCall.result (KernelTensor.realMemory c a before fallback)) before) := by
  unfold runReal
  rw [KernelExecution.source_complete]
  rfl

theorem native_run (profile : NativeTerms.FP32Profile) (c : Host.Invocation) (a : Host.Allocations)
    (valid : Host.Valid c.dims) (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) :
    let logical := KernelExecution.nativeTarget profile c.mathCall valid c.scale.argumentFP32
      (KernelTensor.nativeMemory c a before fallback)
    runNative profile c a before fallback=some (logical,publishBytes a logical before) := by
  unfold runNative
  rw [KernelExecution.native_complete profile c.mathCall valid]
  rfl

def runRealFor (c : Host.Invocation) (a : Host.Allocations) (work : KernelSchedule.Work)
    (before : Nat → Cell ℝ) (fallback : Memory ℝ) : Option (Memory ℝ × (Nat → Cell ℝ)) := do
  let logical ← KernelSchedule.run Real.algebra Real.dotPrimitive c.mathCall work
    (KernelTensor.realMemory c a before fallback)
  return (logical,Host.publish a logical before)

def runNativeFor (profile : NativeTerms.FP32Profile) (c : Host.Invocation) (a : Host.Allocations)
    (work : KernelSchedule.Work) (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) :
    Option ((RegionName → Nat → BitVec 32) × BM.Read) := do
  let logical ← KernelSchedule.runNative profile c.mathCall c.scale.argumentFP32 work
    (KernelTensor.nativeMemory c a before fallback)
  return (logical,publishBytes a logical before)

theorem real_order (c : Host.Invocation) (a : Host.Allocations) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.mathCall work) (before : Nat → Cell ℝ) (fallback : Memory ℝ) :
    runRealFor c a work before fallback=runReal c a before fallback := by
  unfold runRealFor
  rw [KernelSchedule.run_eq Real.algebra Real.dotPrimitive c.mathCall work legal]
  rfl

theorem native_order (profile : NativeTerms.FP32Profile) (c : Host.Invocation) (a : Host.Allocations)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.mathCall work)
    (before : BM.Read) (fallback : RegionName → Nat → BitVec 32) :
    runNativeFor profile c a work before fallback=runNative profile c a before fallback := by
  unfold runNativeFor
  rw [KernelSchedule.native_eq profile c.mathCall c.scale.argumentFP32 work legal]
  rfl

#axiomsClean bytes_at
#axiomsClean address_conformance
#axiomsClean bytes_read
#axiomsClean bytes_frame
#axiomsClean input_bytes
#axiomsClean input_real
#axiomsClean real_run
#axiomsClean native_run
#axiomsClean real_order
#axiomsClean native_order
#specNonCircular Resources avoiding [runReal,runNative,Host.execute,KernelExecution.run,
  KernelExecution.runNative,KernelExecution.nativeTarget,Real.Call.result]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelPhysical
