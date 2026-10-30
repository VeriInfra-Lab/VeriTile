import bench.repairs.tritonbench_g.chunk_delta_fwd.HostMetadata

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Host (nextPowerOfTwo nextPowerOfTwo_ge)
set_option maxHeartbeats 8000000

/-- Actual tensor metadata. Addresses are bytes; ordinary contiguous tensors
use their current data pointer, including a nonzero storage offset. -/
structure Tensor where
  (identity byteBase device : Nat)
  shape : List Nat
  (fp32 ordinary cuda contiguous negative conjugate : Bool)
  deriving DecidableEq

def Tensor.count (t : Tensor) := t.shape.prod

def Tensor.valid (t : Tensor) (shape : List Nat) (device : Nat) : Prop :=
  t.shape=shape ∧ t.device=device ∧ t.fp32=Bool.true ∧ t.ordinary=Bool.true ∧ t.cuda=Bool.true ∧
    t.contiguous=Bool.true ∧ t.negative=Bool.false ∧ t.conjugate=Bool.false
instance (t : Tensor) (shape : List Nat) (device : Nat) : Decidable (t.valid shape device) := by
  unfold Tensor.valid; infer_instance

structure Request where
  (B H T K V BT : Nat)
  builtinBT : Bool
  (key weight value : Tensor)
  (initial final : Option Tensor)

def Request.dims (r : Request) : Host.Dimensions :=
  ⟨r.B,r.H,r.T,r.K,r.V,r.BT,r.H*r.T*r.K,r.T*r.K,r.K,1,r.initial.isSome,r.final.isSome⟩
def Request.params (r : Request) := r.dims.parameters
def Request.inputs (r : Request) := [r.key,r.weight,r.value]++r.initial.toList
def Request.all (r : Request) := r.inputs++r.final.toList

def timeExtent (c : Host.Dimensions) := c.NT*c.BT+ceilDiv c.BT c.BC*c.BC
def valueExtent (c : Host.Dimensions) := ceilDiv c.V c.BV*c.BV
def keyExtent (c : Host.Dimensions) := c.B*c.H*c.T*c.K+c.BK+timeExtent c*c.K
def weightExtent (c : Host.Dimensions) := c.B*c.wBatch+c.H*c.wHead+timeExtent c*c.wTime+c.BK*c.wDim
def valueOffsetExtent (c : Host.Dimensions) := c.B*c.H*c.T*c.V+timeExtent c*c.V+valueExtent c
def historyExtent (c : Host.Dimensions) := c.B*c.H*c.NT*c.K*c.V+c.NT*c.K*c.V+c.BK*c.V+valueExtent c
def stateExtent (c : Host.Dimensions) := c.B*c.H*c.K*c.V+c.BK*c.V+valueExtent c
def formedExtent (c : Host.Dimensions) := keyExtent c+weightExtent c+valueOffsetExtent c+historyExtent c+stateExtent c+1

def Request.scalarExtents (r : Request) : List Nat :=
  [r.B,r.H,r.T,r.K,r.V,r.BT,r.dims.NT,ceilDiv r.V r.dims.BV,
    r.H*r.T*r.K,r.T*r.K,r.T*r.V,r.dims.NT*r.K*r.V,formedExtent r.dims]

def Pointer (t : Tensor) (extent : Nat) : Prop := t.byteBase%4=0 ∧ t.byteBase+4*extent+4<2^63
instance (t : Tensor) (extent : Nat) : Decidable (Pointer t extent) := by unfold Pointer;infer_instance

def Separate (a b : Tensor) : Prop :=
  a.count=0 ∨ b.count=0 ∨ a.byteBase+4*a.count≤b.byteBase ∨ b.byteBase+4*b.count≤a.byteBase
instance (a b : Tensor) : Decidable (Separate a b) := by unfold Separate;infer_instance

def Request.Admitted (r : Request) : Prop :=
  r.builtinBT=Bool.true ∧ 0<r.BT ∧ 0<r.K ∧ r.K≤256 ∧
  (∃ e∈List.range 7,r.dims.BC=2^e) ∧ r.B*r.H≤65535 ∧ ceilDiv r.V r.dims.BV≤65535 ∧
  (∀ n∈r.scalarExtents,n<2^31) ∧
  r.key.valid [r.B,r.H,r.T,r.K] r.key.device ∧
  r.weight.valid [r.B,r.H,r.T,r.K] r.key.device ∧
  r.value.valid [r.B,r.H,r.T,r.V] r.key.device ∧
  (∀ t∈r.initial.toList++r.final.toList,t.valid [r.B,r.H,r.K,r.V] r.key.device) ∧
  (∀ t∈r.all,Pointer t (formedExtent r.dims)) ∧
  (∀ f∈r.final.toList,∀ t∈r.inputs,Separate f t)
instance (r : Request) : Decidable r.Admitted := by unfold Request.Admitted;infer_instance

def Request.check (r : Request) : Bool := decide r.Admitted

theorem Request.check_iff (r : Request) : r.check=Bool.true ↔ r.Admitted := by simp [Request.check]

theorem Request.shape (r : Request) (checked : r.check=Bool.true) : Geometry.ShapeValid r.params := by
  have h:=r.check_iff.mp checked
  have bv:=Host.bv_legal r.dims
  refine ⟨h.2.1,?_,?_,nextPowerOfTwo_ge r.K,rfl⟩
  · change 0<min r.BT r.dims.BV
    exact lt_min h.2.1 (by rcases bv with bv|bv|bv <;> omega)
  · change 0<r.dims.BV
    rcases bv with bv|bv|bv <;> omega

theorem Request.formed (r : Request) (checked : r.check=Bool.true) : formedExtent r.dims<2^31 :=
  (r.check_iff.mp checked).2.2.2.2.2.2.2.1 _ (by simp [Request.scalarExtents])

theorem Request.pointers (r : Request) (checked : r.check=Bool.true) :
    ∀ t∈r.all,Pointer t (formedExtent r.dims) :=
  (r.check_iff.mp checked).2.2.2.2.2.2.2.2.2.2.2.2.1

theorem Request.final_separate (r : Request) (checked : r.check=Bool.true) :
    ∀ f∈r.final.toList,∀ t∈r.inputs,Separate f t :=
  (r.check_iff.mp checked).2.2.2.2.2.2.2.2.2.2.2.2.2

#axiomsClean Request.shape
#axiomsClean Request.formed
#specNonCircular Request.Admitted avoiding [run,runGrid,gridMemory,states]
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
