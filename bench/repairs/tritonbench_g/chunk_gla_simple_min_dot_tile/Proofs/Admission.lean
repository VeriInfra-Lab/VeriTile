import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.NativeControl
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Admission

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Admission
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
abbrev Tensor := VeriTile.Bench.Repairs.ChunkGlaFwd.Admission.Tensor
abbrev Scalar := VeriTile.Bench.Repairs.ChunkGlaFwd.Admission.Scalar

def blocks : List Nat := [16,32,64,128,256,512,1024]
def Sizes (c : Shape) : Prop :=
  4*(c.heads*c.T*c.K)<2^63 ∧ 4*(c.heads*c.T*c.V)<2^63 ∧
  4*(c.heads*c.NT*c.K*c.V)<2^63 ∧ 4*(c.heads*c.T)<2^63
instance (c : Shape) : Decidable (Sizes c) := inferInstanceAs (Decidable (_ ∧ _))

def Launch (c : Shape) : Prop := c.NV<2^31 ∧ c.NT≤65535 ∧ c.heads≤65535
instance (c : Shape) : Decidable (Launch c) := inferInstanceAs (Decidable (_ ∧ _))
def Address (x : Tensor) : Prop := x.byteBase%4=0 ∧ x.byteBase+4*x.shape.prod<2^64
instance (x : Tensor) : Decidable (Address x) := inferInstanceAs (Decidable (_ ∧ _))

structure Request where
  shape : Shape
  history : Tensor
  query : Tensor
  key : Tensor
  value : Tensor
  gate : Tensor
  scale : Scalar

def Request.inputs (r : Request) : List Tensor := [r.history,r.query,r.key,r.value,r.gate]

def Contract (r : Request) : Prop :=
  r.shape.BT∈blocks ∧
  r.query.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T,r.shape.K] ∧
  r.key.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T,r.shape.K] ∧
  r.value.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T,r.shape.V] ∧
  r.history.Valid r.key.device [r.shape.B,r.shape.H,r.shape.NT*r.shape.K,r.shape.V] ∧
  r.gate.Valid r.key.device [r.shape.B,r.shape.H,r.shape.T] ∧ r.scale.Valid ∧
  (r.shape.heads*r.shape.T*r.shape.V≠0 → Sizes r.shape ∧ Launch r.shape ∧ ∀ x∈r.inputs,Address x)
instance (r : Request) : Decidable (Contract r) := inferInstanceAs (Decidable (_ ∧ _))
def check (r : Request) : Bool := decide (Contract r)
theorem check_iff (r : Request) : check r=Bool.true ↔ Contract r := by simp [check]

theorem block_bounds (c : Shape) (h : c.BT∈blocks) : 0<c.BT ∧ c.BT≤1024 := by
  simp only [blocks,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h <;> omega

theorem fits (c : Shape) (h : Sizes c) : Native.Fits c := by
  rcases h with ⟨hq,hv,hh,hg⟩
  intro r
  unfold capacity
  split_ifs <;> omega

/-- The native integer, mask, loop and grid certificate follows from actual
host checks; successful compilation does not provide it as an assumption. -/
theorem accepted (r : Request) (checked : check r=Bool.true)
    (nonempty : r.shape.heads*r.shape.T*r.shape.V≠0) : NativeControl.Accepted r.shape := by
  have contract:=(check_iff r).mp checked
  have finite:=contract.2.2.2.2.2.2.2 nonempty
  refine ⟨fits r.shape finite.1,by omega,?_,?_⟩
  · exact (block_bounds r.shape contract.1).2.trans (by decide)
  · intro q
    obtain ⟨hv,ht,hh⟩:=launch_lt r.shape q
    exact ⟨lt_trans hv finite.2.1.1,lt_of_lt_of_le ht (finite.2.1.2.1.trans (by decide)),
      lt_of_lt_of_le hh (finite.2.1.2.2.trans (by decide))⟩

theorem pointer (x : Tensor) (h : Address x) (i : Nat) (hi : i<x.shape.prod) :
    x.byteBase+4*i<2^64 ∧ (x.byteBase+4*i)%4=0 := by
  have aligned:=h.1
  have room:=h.2
  omega

#axiomsClean accepted
#axiomsClean pointer
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Admission
