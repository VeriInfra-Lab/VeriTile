import bench.repairs.tritonbench_g.flash_attn.Proofs.Bounds

namespace VeriTile.Bench.Repairs.FlashAttn.Checked
set_option maxHeartbeats 2000000

/-- Actual metadata and storage-pointer observations only. -/
structure Tensor where
  tensor : Host.Tensor
  (ordinary strided : Bool)
  (dataPointer storageBase : Nat)
  pointer : dataPointer=storageBase+4*tensor.storageOffset

instance guardDecidable (sizes strides : Fin 4→Nat) (axes : List (Fin 4)) :
    Decidable (Layouts.Guard sizes strides axes) := by
  induction axes with
  | nil=>exact isTrue trivial
  | cons a axes ih=>unfold Layouts.Guard;infer_instance

/-- `axes` is the index order produced by the wrapper's sorted stride scan.
Permutation is an intrinsic sort fact; span comparisons are executed checks. -/
structure SortedAxes where
  axes : List (Fin 4)
  permutation : axes.Perm [0,1,2,3]

def pointerEnvelope (s : Shape) (t : Strides) := envelope s t+512*t.row

def tensorChecks (t : Tensor) (order : SortedAxes) := decide (
  t.ordinary=Bool.true ∧ t.strided=Bool.true ∧
  (0<t.tensor.shape.batches ∧ 0<t.tensor.shape.heads ∧ 0<t.tensor.shape.length ∧ 0<t.tensor.shape.dim) ∧
  (∀ i : Fin 4,Layouts.sizes t.tensor.shape i<2^31) ∧
  (∀ i : Fin 4,0<Layouts.strides t.tensor.strides i ∧ Layouts.strides t.tensor.strides i<2^31) ∧
  (t.tensor.dtypeFP32=Bool.true ∧ t.tensor.cuda=Bool.true ∧ t.tensor.materialized=Bool.true ∧ t.tensor.requiresGrad=Bool.false) ∧
  Layouts.Guard (Layouts.sizes t.tensor.shape) (Layouts.strides t.tensor.strides) order.axes ∧
  4*(t.tensor.storageOffset+Layouts.extent (Layouts.sizes t.tensor.shape) (Layouts.strides t.tensor.strides) order.axes)≤t.tensor.storageBytes ∧
  envelope t.tensor.shape t.tensor.strides<2^31 ∧
  t.dataPointer%4=0 ∧ t.dataPointer+4*(pointerEnvelope t.tensor.shape t.tensor.strides+1)<2^64)

theorem admitted_tensor (t : Tensor) (order : SortedAxes) (h : tensorChecks t order=Bool.true) :
    Host.TensorGuard t.tensor := by
  simp only [tensorChecks,decide_eq_true_eq] at h
  exact ⟨h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,
    ⟨order.axes,order.permutation,h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2.1⟩,h.2.2.2.2.2.2.2.2.1⟩

inductive RawScalar where
  | integer (value : Int)
  | float64 (bits : BitVec 64)
  deriving DecidableEq

/-- Original scalar payload, observed source parameter conversion, and separate
mathematical projection. Python ±0 is never reconstructed from Real. -/
structure ScalarABI where
  raw : RawScalar
  word : Word
  marshal : RawScalar→Word
  marshalled : word=marshal raw
  project : RawScalar→ℝ
  (finite inFP32Range : Bool)

def arguments (q k v : Tensor) (causal : Bool) (scale : ScalarABI)
    (owner : Nat→Nat→Nat→Fin 4) : Host.Arguments :=
  ⟨q.tensor,k.tensor,v.tensor,causal,scale.word,owner⟩

def inputChecks (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI) :=
  tensorChecks q qo && tensorChecks k ko && tensorChecks v vo &&
  scale.finite && scale.inFP32Range && decide (
    k.tensor.shape=q.tensor.shape ∧ v.tensor.shape=q.tensor.shape ∧
    k.tensor.device=q.tensor.device ∧ v.tensor.device=q.tensor.device ∧
    q.tensor.allocation≠k.tensor.allocation ∧ q.tensor.allocation≠v.tensor.allocation ∧ k.tensor.allocation≠v.tensor.allocation ∧
    q.tensor.shape.dim∈[16,32,64,128] ∧ q.tensor.shape.batches*q.tensor.shape.heads≤65535 ∧
    q.tensor.shape.batches*q.tensor.shape.heads*q.tensor.shape.length+127<2^31)

theorem admitted_inputs (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI)
    (h : inputChecks q k v qo ko vo scale=Bool.true) (causal : Bool) (owner : Nat→Nat→Nat→Fin 4) :
    Host.Contract (arguments q k v causal scale owner) := by
  simp only [inputChecks,Bool.and_eq_true,decide_eq_true_eq] at h
  rcases h with ⟨⟨⟨⟨⟨hq,hk⟩,hv⟩,_⟩,_⟩,hs,hvshape,hkdev,hvdev,hqk,hqv,hkv,hd,hgrid,hscratch⟩
  exact ⟨admitted_tensor q qo hq,admitted_tensor k ko hk,admitted_tensor v vo hv,
    ⟨hs,hvshape⟩,⟨hkdev,hvdev⟩,⟨hqk,hqv,hkv⟩,hd,hgrid,hscratch⟩

/-- Actual successful allocation observations, with no supplied guard result. -/
structure Allocations (a : Host.Arguments) (o : Tensor) (scratch : Nat) : Prop where
  outputShape : o.tensor.shape=a.q.shape
  outputDevice : o.tensor.device=a.q.device
  freshOutput : o.tensor.allocation≠a.q.allocation ∧ o.tensor.allocation≠a.k.allocation ∧ o.tensor.allocation≠a.v.allocation
  freshScratch : scratch≠a.q.allocation ∧ scratch≠a.k.allocation ∧ scratch≠a.v.allocation ∧ scratch≠o.tensor.allocation

theorem admitted_allocations (a : Host.Arguments) (o : Tensor) (scratch : Nat)
    (allocation : Allocations a o scratch) (order : SortedAxes) (h : tensorChecks o order=Bool.true) :
    Host.AllocationFacts a o.tensor scratch :=
  ⟨admitted_tensor o order h,allocation.outputShape,allocation.outputDevice,
    allocation.freshOutput,allocation.freshScratch⟩

theorem formed_tensor (t : Tensor) (order : SortedAxes) (h : tensorChecks t order=Bool.true)
    (b head n d : Nat) (hb : b<t.tensor.shape.batches) (hh : head<t.tensor.shape.heads)
    (hn : n≤t.tensor.shape.length+127) (hd : d<t.tensor.shape.dim) (j : Fin 4) :
    (BitVec.ofNat 32 (address t.tensor.strides b head n d)).toInt=(address t.tensor.strides b head n d : Int) ∧
    t.dataPointer%4=0 ∧ t.dataPointer+4*address t.tensor.strides b head n d+j.val<2^64 := by
  have bound:=address_formed_bound t.tensor.shape t.tensor.strides b head n d hb hh hn hd
  have hs:=(admitted_tensor t order h).formed
  simp only [tensorChecks,decide_eq_true_eq] at h
  have hp:=h.2.2.2.2.2.2.2.2.2
  simp only [pointerEnvelope] at hp
  exact ⟨signed32_exact _ (lt_of_le_of_lt bound hs),hp.1,by have := j.isLt;omega⟩

/-- Additional masked compiler prefetch reserve remains bounded without using
any numerical premise. Source signed32 arithmetic uses the tighter bound above. -/
theorem formed_prefetch (t : Tensor) (order : SortedAxes) (h : tensorChecks t order=Bool.true)
    (n : Nat) (hn : n≤pointerEnvelope t.tensor.shape t.tensor.strides) (j : Fin 4) :
    n<2^63 ∧ t.dataPointer+4*n+j.val<2^64 := by
  have guard:=admitted_tensor t order h
  have stride:=guard.positiveStrides 2
  change 0<t.tensor.strides.row ∧ t.tensor.strides.row<2^31 at stride
  have formed:=guard.formed
  simp only [tensorChecks,decide_eq_true_eq] at h
  have hp:=h.2.2.2.2.2.2.2.2.2.2
  unfold pointerEnvelope at hn hp
  have := j.isLt
  constructor <;> omega

def scratchPointerChecks (s : Shape) (base : Nat) :=
  decide (base%4=0 ∧ base+4*(s.batches*s.heads*s.length+128)<2^64)

theorem formed_scratch (a : Host.Arguments) (valid : Host.Contract a) (base : Nat)
    (h : scratchPointerChecks (Host.shape a) base=Bool.true) (l : Lane) (hl : l.Valid (Host.shape a)) (j : Fin 4) :
    (BitVec.ofNat 32 (l.pidHead*a.q.shape.length+(l.pidRow*128+l.rowLane))).toInt=
      (l.pidHead*a.q.shape.length+(l.pidRow*128+l.rowLane) : Int) ∧
    base%4=0 ∧ base+4*(l.pidHead*a.q.shape.length+(l.pidRow*128+l.rowLane))+j.val<2^64 := by
  have hi:=scratch_formed_i32 a valid l hl
  have hr:=row_envelope (Host.shape a) l hl
  have hh:=hl.2.1
  have hb:=Nat.mul_le_mul_right a.q.shape.length (show l.pidHead+1≤a.q.shape.batches*a.q.shape.heads by exact hh)
  simp only [scratchPointerChecks,decide_eq_true_eq,Host.shape] at h
  dsimp [Host.shape] at hr
  exact ⟨signed32_exact _ hi,h.1,by have:=j.isLt; nlinarith [h.2]⟩

#axiomsClean admitted_tensor
#axiomsClean admitted_inputs
#axiomsClean admitted_allocations
#axiomsClean formed_tensor
#axiomsClean formed_prefetch
#axiomsClean formed_scratch
end VeriTile.Bench.Repairs.FlashAttn.Checked
