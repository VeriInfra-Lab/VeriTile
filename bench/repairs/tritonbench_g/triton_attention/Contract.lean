import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.HostSeparate

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native
set_option maxHeartbeats 2000000

structure Geometry where
  (batches heads rows dim : Nat)

def shapeChecks (g : Geometry) := decide (
  0<g.batches ∧ 0<g.heads ∧ 0<g.rows ∧ g.rows%128=0 ∧
  (g.dim=16 ∨ g.dim=32 ∨ g.dim=64 ∨ g.dim=128) ∧
  g.batches*g.heads*g.rows*g.dim≤2147483647 ∧ g.batches*g.heads≤65535)

def checkedShape (g : Geometry) (h : shapeChecks g=Bool.true) : Shape := by
  simp only [shapeChecks,decide_eq_true_eq] at h
  have hn : g.rows/128*128=g.rows := by omega
  exact ⟨g.batches,g.heads,g.rows/128,g.dim,h.1,h.2.1,by omega,h.2.2.2.2.1,
    by simpa only [hn] using h.2.2.2.2.2.1,h.2.2.2.2.2.2⟩

@[simp] theorem checked_rows (g : Geometry) (h : shapeChecks g=Bool.true) : (checkedShape g h).rows=g.rows := by
  simp only [shapeChecks,decide_eq_true_eq] at h
  change g.rows/128*128=g.rows
  omega

inductive RawScalar where
  | integer (value : Int)
  | float64 (bits : BitVec 64)
  deriving DecidableEq

/-- Actual source scalar conversion; the original Python object is retained.
The mathematical projection never reconstructs the ABI word. -/
structure ScalarABI where
  raw : RawScalar
  word : Word
  marshal : RawScalar→Word
  marshalled : word=marshal raw
  project : RawScalar→ℝ

def scaleChecks (s : ScalarABI) := decide (0<s.word.toNat ∧ s.word.toNat<0x7f800000)

/-- Ordinary actual Tensor object/storage and pointer observations. -/
structure Tensor where
  observation : Host.Tensor
  (object storage storageBase storageBytes storageOffset dataPointer numel : Nat)
  width : Nat
  widthPositive : 0<width
  pointer : dataPointer=storageBase+width*storageOffset

def rawTensor (g : Geometry) (device : Nat) (buffer : Buffer) : Host.Tensor :=
  ⟨buffer,[g.batches,g.heads,g.rows,g.dim],[g.heads*g.rows*g.dim,g.rows*g.dim,g.dim,1],
    device,.fp32,Bool.true,Bool.true,Bool.true,Bool.true⟩

/-- This selects the actual FP32 branch of the retained FP32/FP16 wrapper. -/
def tensorChecks (g : Geometry) (device : Nat) (buffer : Buffer) (t : Tensor) :=
  decide (t.observation=rawTensor g device buffer ∧ t.width=4 ∧ t.numel=g.batches*g.heads*g.rows*g.dim)

def inputChecks (g : Geometry) (q k v : Tensor) (scale : ScalarABI) :=
  shapeChecks g && tensorChecks g q.observation.device .q q &&
  tensorChecks g q.observation.device .k k && tensorChecks g q.observation.device .v v && scaleChecks scale

def arguments (q k v : Tensor) (scale : ScalarABI) : Host.Arguments :=
  ⟨q.observation,k.observation,v.observation,scale.word⟩

theorem tensor_admitted (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat) (buffer : Buffer)
    (t : Tensor) (h : tensorChecks g device buffer t=Bool.true) :
    t.observation=Host.tensor (checkedShape g hs) device buffer := by
  simp only [tensorChecks,decide_eq_true_eq] at h
  rw [h.1]
  simp only [rawTensor,Host.tensor,Host.dimensions,Host.strides,checked_rows]
  rfl

theorem inputs_admitted (g : Geometry) (hs : shapeChecks g=Bool.true)
    (q k v : Tensor) (scale : ScalarABI) (hi : inputChecks g q k v scale=Bool.true) :
    Host.InputChecks (checkedShape g hs) (arguments q k v scale) := by
  simp only [inputChecks,Bool.and_eq_true] at hi
  rcases hi with ⟨⟨⟨⟨_,hq⟩,hk⟩,hv⟩,hscale⟩
  simp only [scaleChecks,decide_eq_true_eq] at hscale
  exact ⟨tensor_admitted g hs _ .q q hq,tensor_admitted g hs _ .k k hk,
    tensor_admitted g hs _ .v v hv,hscale.1,hscale.2⟩

def pointerEnvelope (g : Geometry) := g.batches*g.heads*g.rows*g.dim+65536
def pointerChecks (g : Geometry) (t : Tensor) := decide (
  t.width*(t.storageOffset+t.numel)≤t.storageBytes ∧
  t.dataPointer%t.width=0 ∧ t.dataPointer+t.width*(pointerEnvelope g+1)<2^64)

theorem active_bytes (g : Geometry) (t : Tensor) (h : pointerChecks g t=Bool.true)
    (offset : Nat) (ho : offset<t.numel) (j : Nat) (hj : j<t.width) :
    t.storageBase≤t.dataPointer+t.width*offset+j ∧
    t.dataPointer+t.width*offset+j<t.storageBase+t.storageBytes := by
  simp only [pointerChecks,decide_eq_true_eq] at h
  have hm:=Nat.mul_le_mul_left t.width (show offset+1≤t.numel by omega)
  have hp:=t.pointer
  simp only [Nat.mul_add,Nat.mul_one] at hm h hp
  omega

theorem formed_bytes (g : Geometry) (hs : shapeChecks g=Bool.true) (t : Tensor)
    (h : pointerChecks g t=Bool.true) (offset : Nat) (ho : offset≤pointerEnvelope g)
    (j : Nat) (hj : j<t.width) :
    offset<2^63 ∧ t.dataPointer%t.width=0 ∧ t.dataPointer+t.width*offset+j<2^64 := by
  simp only [shapeChecks,decide_eq_true_eq] at hs
  simp only [pointerChecks,decide_eq_true_eq] at h
  have he:=hs.2.2.2.2.2.1
  have hm:=Nat.mul_le_mul_left t.width ho
  simp only [Nat.mul_add,Nat.mul_one] at h
  unfold pointerEnvelope at ho he
  exact ⟨by omega,h.2.1,by omega⟩

def gradientChecks (g : Geometry) (device : Nat) (t : Tensor) := decide (
  t.observation.isTensor=Bool.true ∧ t.observation.strided=Bool.true ∧
  t.observation.shape=[g.batches,g.heads,g.rows,g.dim] ∧ t.observation.device=device ∧
  t.observation.dtype=.fp32 ∧ t.observation.cuda=Bool.true ∧ t.observation.resolved=Bool.true ∧ t.width=4)

theorem gradient_admitted (g : Geometry) (hs : shapeChecks g=Bool.true)
    (q k v t : Tensor) (scale : ScalarABI) (h : gradientChecks g q.observation.device t=Bool.true) :
    Host.GradientChecks (checkedShape g hs) (arguments q k v scale) t.observation := by
  simp only [gradientChecks,decide_eq_true_eq] at h
  refine ⟨h.1,h.2.1,?_,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2.1⟩
  simpa only [Host.dimensions,checked_rows] using h.2.2.1

#axiomsClean checked_rows
#axiomsClean tensor_admitted
#axiomsClean inputs_admitted
#axiomsClean active_bytes
#axiomsClean formed_bytes
#axiomsClean gradient_admitted
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
