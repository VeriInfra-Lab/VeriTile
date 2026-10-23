import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Physical
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Real

namespace VeriTile.Bench.Repairs.Conv2d.Host
open VeriTile.Torch Physical

def outputShape (s : Shape) : List Nat := [s.n,s.outputs,s.oh,s.ow]
def outputStrides (s : Shape) : List Int := [s.outputs*s.oh*s.ow,s.oh*s.ow,s.ow,1]

/-- Ordinary successful torch.empty contract. It supplies allocation identity,
metadata and disjoint storage only; it says nothing about computed values. -/
structure Allocation (c : Call) (out : TensorView) : Prop where
  kind : TensorKind out
  shape : out.tensor.shape=outputShape c.shape
  strides : out.strides=outputStrides c.shape
  device : out.tensor.device=c.input.tensor.device
  bounds : Stored out c.shape.size
  freshInput : out.identity≠c.input.identity
  freshWeight : out.identity≠c.weight.identity
  input : Separated c.input.storageBase c.input.capacity out.pointer (4*c.shape.size)
  weight : Separated c.weight.storageBase c.weight.capacity out.pointer (4*c.shape.size)

def layout (c : Call) (out : TensorView) : Physical.Layout :=
  ⟨c.input.pointer,c.weight.pointer,out.pointer,4*inputWords c.shape,4*weightWords c.shape,c.shape.size⟩

theorem stored_subspan (t : TensorView) (words out length : Nat) (bound : Stored t words)
    (apart : Separated t.storageBase t.capacity out length) :
    Separated t.pointer (4*words) out length := by
  unfold Separated at *
  by_cases empty:words=0
  · simp [empty]
  · obtain ⟨aligned,lo,hi,addr⟩:=bound (by omega)
    omega

theorem layout_fresh (c : Call) (domain : Domain c) (out : TensorView) (a : Allocation c out) :
    Fresh (layout c out) := by
  constructor
  · exact stored_subspan c.input _ _ _ domain.inputStorage a.input
  · exact stored_subspan c.weight _ _ _ domain.weightStorage a.weight

theorem programs_empty (s : Shape) (hn : s.n=0) : programs s=[] := by
  simp [programs,Shape.spatial,hn,blocks]

structure Returned (MemoryType : Type) where
  tensor : TensorView
  currentDevice : Nat
  launches : Nat
  memory : MemoryType

/-- The wrapper returns the freshly allocated Tensor after its direct grid.
The allocator's arbitrary initial output bytes are in `heap`. -/
def invoke (c : Call) (out : TensorView) (p : Native.Instructions) (heap : Heap) : Returned Heap where
  tensor:=out
  currentDevice:=c.callerDevice
  launches:=if c.shape.n=0 then 0 else 1
  memory:=if c.shape.n=0 then heap else execute c.shape (layout c out) p c.flags heap

/-- Ideal Real values at actual element byte addresses. Read-only tensor
aliasing is reflected by both regions observing this same caller carrier. -/
def realInitial (c : Call) (out : TensorView) (heap : Nat → ℝ) : Memory ℝ
  | .input,i => heap (c.input.pointer+4*i)
  | .weight,i => heap (c.weight.pointer+4*i)
  | .output,i => heap (out.pointer+4*i)
  | .other id,i => heap (id+4*i)

noncomputable def invokeReal (c : Call) (out : TensorView) (initial : Memory ℝ) : Returned (Memory ℝ) where
  tensor:=out
  currentDevice:=c.callerDevice
  launches:=if c.shape.n=0 then 0 else 1
  memory:=if c.shape.n=0 then initial else source c.shape realISA c.flags initial

theorem invoke_memory (c : Call) (out : TensorView) (p : Native.Instructions) (heap : Heap) :
    (invoke c out p heap).memory=execute c.shape (layout c out) p c.flags heap := by
  unfold invoke;dsimp only
  split_ifs with empty
  · simp [execute,programs_empty c.shape empty,run]
  · rfl

theorem invokeReal_memory (c : Call) (out : TensorView) (m : Memory ℝ) :
    (invokeReal c out m).memory=source c.shape realISA c.flags m := by
  unfold invokeReal;dsimp only
  split_ifs with empty
  · simp [source,programs_empty c.shape empty,sourcePrograms]
  · rfl

theorem metadata (c : Call) (out : TensorView) (a : Allocation c out) :
    out.tensor.dtype=.floating .fp32 ∧ out.tensor.shape=outputShape c.shape ∧
    out.strides=outputStrides c.shape ∧ out.tensor.device=c.input.tensor.device ∧
    out.identity≠c.input.identity ∧ out.identity≠c.weight.identity :=
  ⟨a.kind.2.1,a.shape,a.strides,a.device,a.freshInput,a.freshWeight⟩

theorem native_read (c : Call) (d : Domain c) (out : TensorView) (a : Allocation c out)
    (p : Native.Instructions) (heap : Heap) (x : Cell) (hx : c.shape.Active x) (k : Fin 4) :
    (invoke c out p heap).memory (out.pointer+4*outputPointer c.shape x+k.val)=
      ordered c.shape (Native.isa p) c.flags (Native.load (observe (layout c out) heap)) x k := by
  rw [invoke_memory]
  exact execute_read c.shape d.valid _ (layout_fresh c d out a) rfl p c.flags heap x hx k

theorem native_frame (c : Call) (d : Domain c) (out : TensorView)
    (p : Native.Instructions) (heap : Heap) (address : Nat)
    (outside : address<out.pointer ∨ out.pointer+4*c.shape.size≤address) :
    (invoke c out p heap).memory address=heap address := by
  rw [invoke_memory]
  exact execute_frame c.shape d.valid _ rfl p c.flags heap address outside

theorem input_frame (c : Call) (d : Domain c) (out : TensorView) (a : Allocation c out)
    (p : Native.Instructions) (heap : Heap) (q : Nat) (hq : q<c.input.capacity) :
    (invoke c out p heap).memory (c.input.storageBase+q)=heap (c.input.storageBase+q) := by
  by_cases empty:c.shape.size=0
  · apply native_frame c d out p heap
    omega
  · apply native_frame c d out p heap
    have h:=a.input
    unfold Separated at h
    omega

theorem weight_frame (c : Call) (d : Domain c) (out : TensorView) (a : Allocation c out)
    (p : Native.Instructions) (heap : Heap) (q : Nat) (hq : q<c.weight.capacity) :
    (invoke c out p heap).memory (c.weight.storageBase+q)=heap (c.weight.storageBase+q) := by
  by_cases empty:c.shape.size=0
  · apply native_frame c d out p heap
    omega
  · apply native_frame c d out p heap
    have h:=a.weight
    unfold Separated at h
    omega

/-- Every observed input byte used by the source is an actual caller byte. -/
theorem input_read (c : Call) (out : TensorView) (heap : Heap) (i : Nat)
    (hi : i < inputWords c.shape) (byte : Fin 4) :
    Native.load (observe (layout c out) heap) .input i byte=
      heap (c.input.pointer+4*i+byte.val) := by
  have hb:4*i+byte.val<4*inputWords c.shape := by have hk:=byte.isLt;omega
  simp only [Native.load,observe,layout,hb,ite_true,Nat.add_assoc]

theorem weight_read (c : Call) (out : TensorView) (heap : Heap) (i : Nat)
    (hi : i<weightWords c.shape) (byte : Fin 4) :
    Native.load (observe (layout c out) heap) .weight i byte=
      heap (c.weight.pointer+4*i+byte.val) := by
  have hb:4*i+byte.val<4*weightWords c.shape := by have hk:=byte.isLt;omega
  simp only [Native.load,observe,layout,hb,ite_true,Nat.add_assoc]

#axiomsClean input_read
#axiomsClean weight_read
#axiomsClean layout_fresh
#axiomsClean invoke_memory
#axiomsClean metadata
#axiomsClean native_read
#axiomsClean native_frame
#axiomsClean input_frame
#axiomsClean weight_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d.Host
