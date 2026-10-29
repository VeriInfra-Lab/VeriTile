import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Host
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.Conv2d.KernelTensor
open VeriTile.Triton VeriTile.Torch
open VeriTile.Torch.FP32 (Tensor TensorLaws)
noncomputable section
set_option maxHeartbeats 2000000

structure View4 where
  (n c h w sn sc sy sx : Nat)

def View4.logical (v : View4) (n c y x : Nat) := ((n*v.c+c)*v.h+y)*v.w+x
def View4.offset (v : View4) (n c y x : Nat) := v.sn*n+v.sc*c+v.sy*y+v.sx*x

def inputView (s : Shape) : View4 := ⟨s.n,s.channels,s.h,s.w,s.sn,s.sc,s.sx,s.sy⟩
def weightView (s : Shape) : View4 := ⟨s.outputs,s.cg,s.kh,s.kw,s.so,s.tc,s.tx,s.ty⟩
def outputView (s : Shape) : View4 := ⟨s.n,s.outputs,s.oh,s.ow,s.outputs*s.oh*s.ow,s.oh*s.ow,s.ow,1⟩

/-- Actual rank-four Tensor and storage binding. Positive arbitrary strides,
nonzero offsets and input/weight aliasing are preserved. These are ordinary
API laws and contain no execution or calculated output property. -/
structure Binding (view : TensorView) (v : View4) where
  tensor : Tensor
  laws : TensorLaws tensor
  metadata : tensor.metadata=view.tensor
  object : tensor.object=view.identity
  shape : tensor.metadata.shape=[v.n,v.c,v.h,v.w]
  strides : view.strides=[v.sn,v.sc,v.sy,v.sx]
  storageBase : tensor.storage.base=view.storageBase
  /-- Complete FP32 elements; untyped storage may contain 1–3 unused tail bytes. -/
  capacity : view.capacity/4=tensor.storage.elements
  indices : ∀ n,n<v.n → ∀ c,c<v.c → ∀ y,y<v.h → ∀ x,x<v.w →
    tensor.physicalIndex (v.logical n c y x)=tensor.storageOffset+v.offset n c y x

theorem numel (b : Binding view v) : b.tensor.metadata.numel=v.n*v.c*v.h*v.w := by
  simp [TensorMetadata.numel,b.shape,Nat.mul_assoc]

theorem logical_bound (v : View4) (n c y x : Nat)
    (hn : n<v.n) (hc : c<v.c) (hy : y<v.h) (hx : x<v.w) :
    v.logical n c y x<v.n*v.c*v.h*v.w := by
  have nc : n*v.c+c<v.n*v.c := by nlinarith
  have nh : (n*v.c+c)*v.h+y<v.n*v.c*v.h := by nlinarith
  unfold View4.logical
  nlinarith

theorem pointer_nat (b : Binding view v) (n c y x : Nat)
    (hn : n<v.n) (hc : c<v.c) (hy : y<v.h) (hx : x<v.w) :
    view.pointer=b.tensor.storage.address b.tensor.storageOffset := by
  have pos : 0<b.tensor.metadata.numel := by
    rw [numel b]
    exact Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (by omega) (by omega)) (by omega)) (by omega)
  have ib:=b.laws.inBounds 0 pos
  have z:=b.indices 0 (by omega) 0 (by omega) 0 (by omega) 0 (by omega)
  simp only [View4.logical,View4.offset,Nat.zero_mul,Nat.mul_zero,Nat.zero_add,Nat.add_zero] at z
  rw [z] at ib
  have span:=b.tensor.storage.spanFits
  have fits : b.tensor.storage.address b.tensor.storageOffset<2^64 := by
    unfold FP32.Storage.address
    omega
  change view.tensor.dataPtr.toNat=_
  rw [←b.metadata]
  change b.tensor.dataPtr.toNat=_
  rw [b.laws.pointer pos,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]

theorem address (b : Binding view v) (n c y x : Nat)
    (hn : n<v.n) (hc : c<v.c) (hy : y<v.h) (hx : x<v.w) :
    view.pointer+4*v.offset n c y x=b.tensor.storage.address (b.tensor.physicalIndex (v.logical n c y x)) := by
  rw [pointer_nat b n c y x hn hc hy hx,b.indices n hn c hc y hy x hx]
  unfold FP32.Storage.address
  ring

def realPoint (heap : Nat → ℝ) (b : Binding view v) (n c y x : Nat) : ℝ :=
  heap (b.tensor.storage.address (b.tensor.physicalIndex (v.logical n c y x)))
def nativePoint (heap : Nat → FP.Value .fp32) (b : Binding view v) (n c y x : Nat) :=
  FP32.logicalValue heap b.tensor b.laws (v.logical n c y x)

theorem real_read (heap : Nat → ℝ) (b : Binding view v) (n c y x : Nat)
    (hn : n<v.n) (hc : c<v.c) (hy : y<v.h) (hx : x<v.w) :
    heap (view.pointer+4*v.offset n c y x)=realPoint heap b n c y x := by
  rw [address b n c y x hn hc hy hx]
  rfl

theorem native_read (heap : Nat → FP.Value .fp32) (b : Binding view v)
    (kind : TensorKind view) (n c y x : Nat)
    (hn : n<v.n) (hc : c<v.c) (hy : y<v.h) (hx : x<v.w) :
    heap (view.pointer+4*v.offset n c y x)=nativePoint heap b n c y x := by
  rw [address b n c y x hn hc hy hx]
  simp [nativePoint,FP32.logicalValue,b.metadata,kind.2.2.2.1]

/-- One prepared kernel launch with the already admitted call and actual
input, weight and output Tensors. Allocation metadata is reused as a boundary;
no new wrapper or allocator implementation is introduced. -/
structure Prepared where
  call : Call
  admitted : checkCall call=.ok ()
  output : TensorView
  allocation : Host.Allocation call output
  input : Binding call.input (inputView call.shape)
  weight : Binding call.weight (weightView call.shape)
  result : Binding output (outputView call.shape)

def Prepared.domain (p : Prepared) : Domain p.call := (checkCall_iff p.call).mp p.admitted

#axiomsClean logical_bound
#axiomsClean pointer_nat
#axiomsClean address
#axiomsClean native_read
end
end VeriTile.Bench.Repairs.Conv2d.KernelTensor
