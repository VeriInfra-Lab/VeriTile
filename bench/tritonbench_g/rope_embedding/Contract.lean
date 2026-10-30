import VeriTile.Torch.FP32
import VeriTile.Triton.Float.ScalarBackend
import Mathlib.Data.Nat.Log
import Mathlib.Tactic.Ring
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton
noncomputable section

structure Request where
  (batches heads sequence dimension : Nat)
  deriving DecidableEq
def Request.rows (r : Request) := r.batches*r.sequence
def Request.half (r : Request) := r.dimension/2
def Request.block (r : Request) := if r.half=0 then 0 else 2^Nat.clog 2 r.half
def Request.groups (r : Request) := (r.heads+3)/4
def Request.warps (r : Request) := if 32768≤r.block then 32 else
  if 8192≤r.block then 16 else if 2048≤r.block then 8 else 4
def Request.size (r : Request) := r.rows*r.heads*r.dimension
def Request.offset (r : Request) (row head c : Nat) := (row*r.heads+head)*r.dimension+c
def Request.external (r : Request) (b head m c : Nat) := ((b*r.heads+head)*r.sequence+m)*r.dimension+c

/-- Empty payloads and dimensions below two submit no kernel. -/
def Request.Launched (r : Request) (row group : Nat) : Prop :=
  0 < r.half ∧ row < r.rows ∧ group < r.groups

namespace Spec
/-- Fixed coefficients need not lie on a unit circle. -/
def pair (backward : Bool) (a b cosine sine : ℝ) : ℝ×ℝ :=
  if backward then (a*cosine+b*sine,b*cosine-a*sine)
  else (a*cosine-b*sine,b*cosine+a*sine)
def result (r : Request) (backward : Bool) (q : Nat→ℝ) (cosine sine : Nat→ℝ)
    (row head c : Nat) : ℝ :=
  if c<2*r.half then
    let lane := if c<r.half then c else c-r.half
    let values := pair backward (q (r.offset row head lane))
      (q (r.offset row head (lane+r.half)))
      (cosine (row%r.sequence*r.half+lane)) (sine (row%r.sequence*r.half+lane))
    if c<r.half then values.1 else values.2
  else q (r.offset row head c)
theorem adjoint (a b u v c s : ℝ) :
    (pair Bool.false a b c s).1*u+(pair Bool.false a b c s).2*v=
      a*(pair Bool.true u v c s).1+b*(pair Bool.true u v c s).2 := by
  simp only [pair,Bool.false_eq_true,if_false,if_true]
  ring
end Spec

/-- Tensor metadata at the kernel boundary, after the original preparation.
Q is the packed [B*S,H*D] view; coefficient rows use the original squeeze
shape, including its [L] result when D/2=1. No allocation or launch model. -/
structure KernelMetadata where
  request : Request
  (q cosine sine : VeriTile.Torch.TensorMetadata)
  (cosineRows sineRows : Nat)

def payloadCheck (t : VeriTile.Torch.TensorMetadata) : Bool :=
  t.isFP32 && t.isStrided && t.isCuda && t.isContiguous && !t.isNeg

def tableShape (half rows : Nat) (shape : List Nat) : Prop :=
  shape=[rows,half] ∨ (half=1 ∧ shape=[rows])
instance (half rows : Nat) (shape : List Nat) : Decidable (tableShape half rows shape) :=
  inferInstanceAs (Decidable (_∨_))

def outputLast (r : Request) :=
  ((r.rows-1)*r.heads+(r.heads-1))*r.dimension+(r.block-1)+r.half
def tableLast (r : Request) := (r.sequence-1)*r.half+(r.block-1)

def separateViews (a b : VeriTile.Torch.TensorMetadata) : Prop :=
  a.numel=0 ∨ b.numel=0 ∨ a.dataPtr.toNat+4*a.numel≤b.dataPtr.toNat ∨
    b.dataPtr.toNat+4*b.numel≤a.dataPtr.toNat
instance (a b : VeriTile.Torch.TensorMetadata) : Decidable (separateViews a b) :=
  inferInstanceAs (Decidable (_∨_∨_∨_))

def KernelMetadata.check (m : KernelMetadata) : Bool :=
  let r := m.request
  [m.q,m.cosine,m.sine].all payloadCheck &&
  decide (m.q.shape=[r.rows,r.heads*r.dimension] ∧
    tableShape r.half m.cosineRows m.cosine.shape ∧
    tableShape r.half m.sineRows m.sine.shape ∧
    r.sequence≤m.cosineRows ∧ r.sequence≤m.sineRows ∧
    m.cosine.device=m.q.device ∧ m.sine.device=m.q.device ∧
    separateViews m.q m.cosine ∧ separateViews m.q m.sine) &&
  decide (r.block≤65536 ∧ r.batches<2^31 ∧ r.heads+3<2^31 ∧
    r.sequence<2^31 ∧ r.dimension<2^31 ∧ r.rows<2^31 ∧
    r.heads*r.dimension<2^31 ∧ r.groups<2^31) &&
  decide (r.size≠0 ∧ r.half≠0 →
    outputLast r<2^31 ∧ tableLast r<2^31 ∧
    m.q.dataPtr.toNat+4*outputLast r+4≤2^64 ∧
    m.cosine.dataPtr.toNat+4*tableLast r+4≤2^64 ∧
    m.sine.dataPtr.toNat+4*tableLast r+4≤2^64) &&
  [m.q,m.cosine,m.sine].all (fun t => decide (t.numel=0 ∨ t.dataPtr.toNat%4=0))

#specNonCircular Spec.result avoiding [FP.Structural.exec]
#axiomsClean Spec.adjoint
end
end VeriTile.Bench.Repairs.RopeEmbedding
