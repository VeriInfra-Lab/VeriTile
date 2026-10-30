import bench.tritonbench_g.chunk_gla_fwd.Proofs.PhysicalReturned

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Admission
open VeriTile Triton
set_option maxHeartbeats 4000000

structure Dimensions where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  deriving DecidableEq

def blockSizes : List Nat := [16,32,64,128,256,512,1024]
def ShapeValid (d : Dimensions) : Prop := 0<d.K ∧ d.BT∈blockSizes
instance (d : Dimensions) : Decidable (ShapeValid d) := inferInstanceAs (Decidable (_ ∧ _))
def checkShape (d : Dimensions) : Bool := decide (ShapeValid d)

theorem checkShape_iff (d : Dimensions) : checkShape d=Bool.true ↔ ShapeValid d := by simp [checkShape]

def Dimensions.checked (d : Dimensions) (h : ShapeValid d) : Host.Dimensions where
  B:=d.B
  H:=d.H
  T:=d.T
  K:=d.K
  V:=d.V
  BT:=d.BT
  k_pos:=h.1
  bt_power:=by
    have hm:=h.2
    simp only [blockSizes,List.mem_cons,List.not_mem_nil,or_false] at hm
    rcases hm with h|h|h|h|h|h|h
    · exact ⟨0,by simpa [h]⟩
    · exact ⟨1,by simpa [h]⟩
    · exact ⟨2,by simpa [h]⟩
    · exact ⟨3,by simpa [h]⟩
    · exact ⟨4,by simpa [h]⟩
    · exact ⟨5,by simpa [h]⟩
    · exact ⟨6,by simpa [h]⟩

inductive Mode where | producer | consumer deriving DecidableEq

structure Tensor where
  isTensor : Bool
  strided : Bool
  fp32 : Bool
  cuda : Bool
  contiguous : Bool
  negative : Bool
  conjugate : Bool
  device : Nat
  shape : List Nat
  byteBase : Nat
  deriving DecidableEq

structure Scalar where
  builtinNumber : Bool
  conversionSucceeded : Bool
  finite : Bool
  withinFP32 : Bool
  runtimeWord : BitVec 32
  deriving DecidableEq

def Tensor.Valid (x : Tensor) (device : Nat) (shape : List Nat) : Prop :=
  x.isTensor=Bool.true ∧ x.strided=Bool.true ∧ x.fp32=Bool.true ∧ x.cuda=Bool.true ∧
  x.contiguous=Bool.true ∧ x.negative=Bool.false ∧ x.conjugate=Bool.false ∧ x.device=device ∧ x.shape=shape
instance (x : Tensor) (device : Nat) (shape : List Nat) : Decidable (x.Valid device shape) :=
  inferInstanceAs (Decidable (_ ∧ _))
def Scalar.Valid (x : Scalar) : Prop :=
  x.builtinNumber=Bool.true ∧ x.conversionSucceeded=Bool.true ∧ x.finite=Bool.true ∧ x.withinFP32=Bool.true
instance (x : Scalar) : Decidable x.Valid := inferInstanceAs (Decidable (_ ∧ _))

def extents (c : Host.Dimensions) : List Nat :=
  [c.heads*c.T*c.K,c.heads*c.T*c.BT,c.NK*c.heads*c.T*16,
   c.heads*c.T*c.V,c.heads*c.NT*c.K*c.V]
def bound (c : Host.Dimensions) : Nat :=
  (extents c).foldl max 0+c.BT*max (max (max c.K c.V) c.BT) 16+256*max c.V 1+256

def grids (mode : Mode) (c : Host.Dimensions) : List (Nat×Nat×Nat) :=
  match mode with
  | .producer => [(c.NT,c.NC*c.NC,c.heads),(c.NT,c.NC,c.heads)] ++
      if c.K≤256 then [] else [(c.NK,c.NT*c.NC,c.heads)]
  | .consumer => [(Geometry.ceilDiv c.V c.BV,c.NT,c.heads)]
def empty (mode : Mode) (c : Host.Dimensions) : Bool :=
  match mode with
  | .producer => c.heads*c.T==0
  | .consumer => c.heads*c.T*c.V==0

def Finite (mode : Mode) (c : Host.Dimensions) : Prop :=
  (∀ n∈[c.B,c.H,c.T,c.K,c.V,c.BT,c.NT,c.NK,c.heads],n<2^31) ∧
  bound c<2^31 ∧ ∀ g∈grids mode c,g.1<2^31 ∧ g.2.1≤65535 ∧ g.2.2≤65535
instance (mode : Mode) (c : Host.Dimensions) : Decidable (Finite mode c) :=
  inferInstanceAs (Decidable (_ ∧ _))
def Tensor.AddressValid (x : Tensor) (c : Host.Dimensions) : Prop :=
  x.byteBase%4=0 ∧ x.byteBase+4*bound c<2^64
instance (x : Tensor) (c : Host.Dimensions) : Decidable (x.AddressValid c) :=
  inferInstanceAs (Decidable (_ ∧ _))

structure Request where
  dims : Dimensions
  mode : Mode
  query : Tensor
  key : Tensor
  gate : Tensor
  value : Tensor
  attention : Tensor
  history : Tensor
  scale : Scalar

def inputs (r : Request) : List Tensor :=
  match r.mode with
  | .producer => [r.query,r.key,r.gate]
  | .consumer => [r.query,r.value,r.gate,r.attention,r.history]

def TensorContract (r : Request) (c : Host.Dimensions) : Prop :=
  r.query.Valid r.query.device [c.B,c.H,c.T,c.K] ∧
  r.gate.Valid r.query.device [c.B,c.H,c.T,c.K] ∧ r.scale.Valid ∧
  match r.mode with
  | .producer => r.key.Valid r.query.device [c.B,c.H,c.T,c.K]
  | .consumer => r.value.Valid r.query.device [c.B,c.H,c.T,c.V] ∧
      r.attention.Valid r.query.device [c.B,c.H,c.T,c.BT] ∧
      r.history.Valid r.query.device [c.B,c.H,c.NT*c.K,c.V]
instance (r : Request) (c : Host.Dimensions) : Decidable (TensorContract r c) := by
  unfold TensorContract
  cases r.mode <;> infer_instance

def Contract (r : Request) (c : Host.Dimensions) : Prop :=
  TensorContract r c ∧ (empty r.mode c=Bool.false → Finite r.mode c ∧ ∀ x∈inputs r,x.AddressValid c)
instance (r : Request) (c : Host.Dimensions) : Decidable (Contract r c) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- The proof-carrying dimensions are constructed from the same finite block
membership and K guard as the executable Python wrapper. -/
def admit (r : Request) : Except String {c : Host.Dimensions // Contract r c} :=
  if h : ShapeValid r.dims then
    let c:=r.dims.checked h
    if ok : Contract r c then .ok ⟨c,ok⟩ else .error "tensor, scale, grid, or address guard"
  else .error "shape guard"

theorem admitted (r : Request) (a : {c : Host.Dimensions // Contract r c})
    (_ : admit r=.ok a) : TensorContract r a.val ∧
      (empty r.mode a.val=Bool.false → Finite r.mode a.val ∧ ∀ x∈inputs r,x.AddressValid a.val) := a.property

theorem checked_address (x : Tensor) (c : Host.Dimensions) (h : x.AddressValid c)
    (offset : Nat) (ho : offset≤bound c) :
    x.byteBase+4*offset<2^64 ∧ (x.byteBase+4*offset)%4=0 := by
  have aligned:=h.1
  have room:=h.2
  constructor
  · omega
  · omega

/-- Allocation lengths and identities are ordinary tensor API facts. The
finite pointer guard supplies the byte bound; no source completion is assumed. -/
def allocation (x : Tensor) (length : Nat) : FPStorage.Allocation := ⟨x.byteBase/4,length⟩
theorem allocation_finite (x : Tensor) (c : Host.Dimensions) (h : x.AddressValid c)
    (length : Nat) (hl : length≤bound c) :
    (allocation x length).base+(allocation x length).length<2^62 := by
  have room:=h.2
  simp only [allocation]
  omega

#axiomsClean admitted
#axiomsClean checked_address
#axiomsClean allocation_finite
end VeriTile.Bench.Repairs.ChunkGlaFwd.Admission
