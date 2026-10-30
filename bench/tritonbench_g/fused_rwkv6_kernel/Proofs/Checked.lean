import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.CheckedIndices
import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Checked
noncomputable section
open VeriTile.Torch

structure Tensor where
  handle : Nat
  metadata : TensorMetadata
  strides : List Nat
  storageBase : Nat
  storageOffset : Nat
  exactTensor : Bool
  conjugate : Bool
  deriving DecidableEq

def Tensor.Valid (x : Tensor) : Prop :=
  x.exactTensor=true ∧ x.metadata.layout=.strided ∧ x.metadata.dtype=.floating .fp32 ∧
  x.metadata.device.kind="cuda" ∧ x.metadata.isContiguous=true ∧
  x.metadata.isNeg=false ∧ x.conjugate=false

/-- Python's exact built-in int/float classification; the scalar value is saved
unchanged. Its FP32 launch conversion is a local scalar runtime boundary. -/
inductive PythonNumber where
  | integer (value : Int)
  | float (bits : BitVec 64)
  deriving DecidableEq

/-- Exact helper sentinel comparison; Python's `K**-0.5` result is supplied
as the ordinary host scalar observation. IEEE -1.0 and integer -1 both select it. -/
def selectScale (requested defaultScale : PythonNumber) : PythonNumber :=
  if requested=.integer (-1) ∨ requested=.float 0xbff0000000000000 then defaultScale else requested

structure Inputs where
  tensor : Region → Tensor
  initial : Option Tensor
  boolArguments : Bool
  numberArgument : Bool

def inputRegions (c : Shape) : List Region :=
  [.q,.k,.v,.w,.u] ++ if c.initial then [.h0] else []

def tensorShape (c : Shape) : Region → List Nat
  | .q | .k | .w => [c.B,c.H,c.T,c.K]
  | .v | .result => [c.B,c.H,c.T,c.V]
  | .u => [c.H,c.K]
  | .h0 | .ht => [c.B,c.H,c.K,c.V]
  | .o => [c.NK,c.B,c.H,c.T,c.V]
  | .outside => []

def negative (c : Shape) : Region → Nat
  | .q | .k | .w => if c.reverse then c.K else 0
  | .v | .o => if c.reverse then c.V else 0
  | _ => 0

def PointerGuard (c : Shape) (r : Region) (x : Tensor) : Prop :=
  x.metadata.numel=0 ∨ (x.metadata.dataPtr.toNat%4=0 ∧
    4*negative c r ≤ x.metadata.dataPtr.toNat ∧
    x.metadata.dataPtr.toNat+4*(indexLimit c+1)<2^64)

def AddressFits (x : Tensor) (offset : Int) (j : Fin 4) : Prop :=
  0≤(x.metadata.dataPtr.toNat:Int)+4*offset+j.val ∧
    (x.metadata.dataPtr.toNat:Int)+4*offset+j.val<2^64

/-- Checked room includes all four bytes and negative reverse increments. -/
theorem formed_address (c : Shape) (r : Region) (x : Tensor) (guard : PointerGuard c r x)
    (active : 0<x.metadata.numel) (offset : Int) (low : -(negative c r:Int)≤offset)
    (high : offset ≤ indexLimit c) (j : Fin 4) : AddressFits x offset j := by
  rcases guard with empty|⟨_,lo,hi⟩
  · omega
  · unfold AddressFits
    have hj := j.isLt
    constructor <;> omega

def Geometry (c : Shape) : Prop :=
  0<c.B ∧ 0<c.H ∧ 0<c.K ∧ 0<c.V ∧
  c.keyPitch=c.T*c.K ∧ c.valuePitch=c.T*c.V ∧
  c.BK=min (2^Nat.clog 2 c.K) 32 ∧ c.BV=min (2^Nat.clog 2 c.V) 32 ∧
  c.NV<2^31 ∧ c.NK≤65535 ∧ c.Z≤65535 ∧ indexLimit c<2^31

theorem geometry_iff (c : Shape) : Geometry c ↔ ShapeContract c := by
  constructor
  · rintro ⟨b,h,k,v,kp,vp,bk,bv,gx,gy,gz,limit⟩
    have bp : 0<c.BK := by rw [bk];exact lt_min (pow_pos (by omega) _) (by omega)
    have vp' : 0<c.BV := by rw [bv];exact lt_min (pow_pos (by omega) _) (by omega)
    exact ⟨⟨b,h,k,v,bp,vp'⟩,kp,vp,bk,bv,by omega,gy,gz,limit⟩
  · intro h
    exact ⟨h.sizes.B,h.sizes.H,h.sizes.K,h.sizes.V,h.keyPitch,h.valuePitch,
      h.bkSource,h.bvSource,by have := h.gridX;omega,h.gridY,h.gridZ,h.limit⟩

instance (x : Tensor) : Decidable x.Valid := by unfold Tensor.Valid;infer_instance
instance (c : Shape) : Decidable (Geometry c) := by unfold Geometry;infer_instance
instance (c : Shape) (r : Region) (x : Tensor) : Decidable (PointerGuard c r x) := by
  unfold PointerGuard;infer_instance

/-- Finite literal wrapper guards. No computed output is inspected. -/
def Guards (c : Shape) (x : Inputs) : Prop :=
  Geometry c ∧ x.boolArguments=true ∧ x.numberArgument=true ∧
  x.initial=(if c.initial then some (x.tensor .h0) else none) ∧
  ∀ r∈inputRegions c,(x.tensor r).Valid ∧ (x.tensor r).metadata.shape=tensorShape c r ∧
    (x.tensor r).metadata.device=(x.tensor .q).metadata.device ∧ PointerGuard c r (x.tensor r)

instance (c : Shape) (x : Inputs) : Decidable (Guards c x) := by
  unfold Guards;infer_instance

def admit (c : Shape) (x : Inputs) : Bool := decide (Guards c x)

theorem admit_iff (c : Shape) (x : Inputs) : admit c x=true ↔ Guards c x := by
  simp only [admit,decide_eq_true_eq]

def domain (c : Shape) (x : Inputs) (h : admit c x=true) : Domain c :=
  ((geometry_iff c).mp ((admit_iff c x).mp h).1).domain c

/-- Original direct Tensor descriptors, including the data-pointer offset.
The ordinary contiguous Tensor API identifies these physical cells; no copy or
source-result equation is a field. Singleton unused strides may be arbitrary. -/
structure DirectInputs (c : Shape) (x : Inputs) (base : Region → Nat) : Prop where
  pointers : ∀ r∈inputRegions c,(x.tensor r).metadata.dataPtr.toNat=base r
  storage : ∀ r∈inputRegions c,0<(x.tensor r).metadata.numel →
    (x.tensor r).metadata.dataPtr.toNat=(x.tensor r).storageBase+4*(x.tensor r).storageOffset

structure Allocations (c : Shape) (x : Inputs) (base : Region → Nat) where
  scratch : Tensor
  output : Tensor
  terminal : Tensor
  scratchMetadata : scratch.Valid ∧ scratch.metadata.shape=tensorShape c .o ∧
    scratch.metadata.device=(x.tensor .q).metadata.device ∧ scratch.metadata.dataPtr.toNat=base .o
  outputMetadata : output.Valid ∧ output.metadata.shape=tensorShape c .result ∧
    output.metadata.device=(x.tensor .q).metadata.device ∧ output.metadata.dataPtr.toNat=base .result
  terminalMetadata : c.final=true → terminal.Valid ∧ terminal.metadata.shape=tensorShape c .ht ∧
    terminal.metadata.device=(x.tensor .q).metadata.device ∧ terminal.metadata.dataPtr.toNat=base .ht
  scratchChecked : PointerGuard c .o scratch
  terminalChecked : c.final=true → PointerGuard c .ht terminal

def saved (x : Inputs) : List (Option Tensor) :=
  [some (x.tensor .q),some (x.tensor .k),some (x.tensor .v),some (x.tensor .w),some (x.tensor .u),x.initial]

structure Returned (M : Type) where
  output : Tensor
  finalState : Option Tensor
  saved : List (Option Tensor)
  savedScale : PythonNumber
  savedReverse : Bool
  callerDevice : Device
  memory : M

def returned {M : Type} (c : Shape) (x : Inputs) (base : Region → Nat)
    (a : Allocations c x base) (pythonScale : PythonNumber) (caller : Device) (memory : M) : Returned M :=
  ⟨a.output,if c.final then some a.terminal else none,saved x,pythonScale,c.reverse,caller,memory⟩

#axiomsClean formed_address
#axiomsClean geometry_iff
#axiomsClean admit_iff
#axiomsClean domain
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Checked
