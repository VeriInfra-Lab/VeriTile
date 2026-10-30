import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.TensorGeometry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
noncomputable section

def laneRole {m d : Nat} : Lane m d → Role
  | .center _ => .center | .denominator _ => .denominator | .output .. => .output

def laneIndex (c : Call) (idx : GridIndex c.grid) : Lane c.BM c.D → Nat
  | .center row | .denominator row => c.carryAddress idx row.val
  | .output row dim => index4 c.H c.rounded c.D (c.hz idx/c.H) (c.hz idx%c.H) (c.row idx row.val) dim.val

/-- All BM-padded stores map to bounded logical elements of the actual
supplied O/M/L Tensors. No coordinate or numerical result is an assumption. -/
theorem logical_lane (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) (idx : GridIndex c.grid) (lane : Lane c.BM c.D) :
    laneIndex c idx lane<(r (laneRole lane)).tensor.metadata.numel ∧
    Ownership.address c idx lane=(region (laneRole lane),
      offset (r (laneRole lane)).tensor.metadata.shape (r (laneRole lane)).strides (laneIndex c idx lane)) := by
  have valid := (Contract.safety a.checked).valid
  have batch := batch_bound c valid idx
  have head := head_bound c valid idx
  have hz : c.hz idx/c.H*c.H+c.hz idx%c.H=c.hz idx := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod (c.hz idx) c.H
  cases lane with
  | center row =>
    have geometry := carry_index c A p r a .center (Or.inl rfl)
      (c.hz idx/c.H) (c.hz idx%c.H) (c.row idx row.val) batch head (row_bound c valid idx row)
    rw [hz] at geometry
    exact ⟨geometry.1,congrArg (fun off => (region .center,off)) geometry.2.symm⟩
  | denominator row =>
    have geometry := carry_index c A p r a .denominator (Or.inr rfl)
      (c.hz idx/c.H) (c.hz idx%c.H) (c.row idx row.val) batch head (row_bound c valid idx row)
    rw [hz] at geometry
    exact ⟨geometry.1,congrArg (fun off => (region .denominator,off)) geometry.2.symm⟩
  | output row dim =>
    have geometry := output_index c A p r a (c.hz idx/c.H) (c.hz idx%c.H)
      (c.row idx row.val) dim.val batch head (row_bound c valid idx row) dim.isLt
    exact ⟨geometry.1,congrArg (fun off => (region .output,off)) geometry.2.symm⟩

theorem padded_tensor_readback (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (final : RegionName → Nat → Bytes.Word) (before : Bytes.Memory)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) (live : active c lane) :
    Torch.FP32.logicalValue
      (fun addr => ⟨VeriTile.Bench.Repairs.LightningAttention.KernelOnly.load32 (Bytes.publish c A final before) addr⟩)
      (r (laneRole lane)).tensor (a.laws (laneRole lane)).tensor (laneIndex c idx lane)=
      ⟨final (Ownership.address c idx lane).1 (Ownership.address c idx lane).2⟩ :=
  Bytes.tensor_readback c A p r a closed final before (laneRole lane) (laneIndex c idx lane)
    (logical_lane c A p r a idx lane).1 idx lane live (logical_lane c A p r a idx lane).2

#axiomsClean logical_lane
#axiomsClean padded_tensor_readback
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
