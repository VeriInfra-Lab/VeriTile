import bench.repairs.tritonbench_g.chunk_retention_ops.Contract
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.HostABI
import Mathlib.Tactic
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Contract

theorem tensorCheck_iff (t : Tensor) (shape : List Nat) (device : VeriTile.Torch.Device)
    (canonical contiguous : Bool) :
    tensorCheck t shape device canonical contiguous=Bool.true ↔
      TensorContract t shape device canonical contiguous := by
  cases canonical <;> cases contiguous <;>
    simp [tensorCheck,TensorContract,Backing,Bool.and_eq_true,and_assoc]

theorem shapeCheck_iff (d : Dimensions) : shapeCheck d=Bool.true ↔ ShapeContract d := by
  simp only [shapeCheck,Bool.and_eq_true,decide_eq_true_eq,max_le_iff,max_lt_iff,lt_min_iff]
  have h : max (d.B*d.H*ceil64 d.T*d.K*d.V)
      (ceil64 d.K*d.B*d.H*d.T*d.V)*4<2^63 ↔
      d.B*d.H*ceil64 d.T*d.K*d.V*4<2^63 ∧
      ceil64 d.K*d.B*d.H*d.T*d.V*4<2^63 := by
    omega
  rw [h]
  unfold ShapeContract
  tauto

theorem scaleCheck_iff (s : Scalar) : scaleCheck s=Bool.true ↔ ScaleContract s := by
  simp [scaleCheck,ScaleContract,Bool.and_eq_true,and_assoc]

theorem forwardCheck_iff (a : Forward) : forwardCheck a=Bool.true ↔ ForwardContract a := by
  simp [forwardCheck,ForwardContract,tensorCheck_iff,shapeCheck_iff,scaleCheck_iff,
    Bool.and_eq_true,and_assoc]

theorem backwardCheck_iff (a : Backward) : backwardCheck a=Bool.true ↔ BackwardContract a := by
  simp [backwardCheck,BackwardContract,forwardCheck_iff,tensorCheck_iff,
    Bool.and_eq_true,and_assoc]

theorem no_overlap_iff (a b : Tensor) : overlapCheck a b=Bool.false ↔ disjoint a b := by
  simp only [overlapCheck,decide_eq_false_iff_not,disjoint]
  omega

theorem tileWidth_cases (n : Nat) : tileWidth n=16 ∨ tileWidth n=32 ∨ tileWidth n=64 := by
  unfold tileWidth
  split_ifs <;> simp

def toShape (d : Dimensions) (h : ShapeContract d) : Whole001.Shape where
  B := d.B
  H := d.H
  T := d.T
  K := d.K
  V := d.V
  BK := tileWidth d.K
  BV := tileWidth d.V
  keyPositive := h.2.2.2.1
  valuePositive := h.2.2.2.2.1
  keyTile := tileWidth_cases d.K
  valueTile := tileWidth_cases d.V

theorem chosen_blocks (n : Nat) (hn : 0<n) :
    (n+tileWidth n-1)/tileWidth n=ceil64 n := by
  unfold tileWidth ceil64
  split_ifs <;> omega

theorem chosen_padding (n : Nat) (hn : 0<n) :
    (n+tileWidth n-1)/tileWidth n*tileWidth n≤ceil64 n*64 := by
  rw [chosen_blocks n hn]
  apply Nat.mul_le_mul_left
  rcases tileWidth_cases n with h|h|h <;> omega

theorem launchABI (d : Dimensions) (h : ShapeContract d) : Whole001.HostABI.LaunchABI (toShape d h) := by
  rcases h with ⟨hb,hh,ht,hk,hv,hbh,hnt,hnv,hkt,hvt,hist,hkp,hvp,hist64,part64⟩
  have hpadk := chosen_padding d.K hk
  have hpadv := chosen_padding d.V hv
  have hkey := chosen_blocks d.K hk
  have hval := chosen_blocks d.V hv
  constructor
  · constructor
    · exact hb
    · exact hh
    · exact ht
    · exact hbh
    · exact hkt
    · exact hvt
    · simpa [Whole001.Shape.historyHead,Whole001.Shape.stateHead,Whole001.Shape.chunks,toShape,ceil64,Nat.mul_assoc] using hist
    · exact lt_of_le_of_lt hpadk hkp
    · exact lt_of_le_of_lt hpadv hvp
    · simpa [Whole001.Shape.historySize,Whole001.Shape.historyHead,Whole001.Shape.stateHead,
        Whole001.Shape.heads,Whole001.Shape.chunks,toShape,ceil64,Nat.mul_assoc] using hist64
    · simpa [Whole001.Shape.partialSize,Whole001.Shape.keyBlocks,Whole001.Shape.valueSize,
        Whole001.Shape.valueHead,Whole001.Shape.heads,toShape,hkey,Nat.mul_assoc] using part64
  · exact hnt
  · simpa [Whole001.Shape.valueBlocks,toShape,hval] using hnv

#axiomsClean tensorCheck_iff
#axiomsClean forwardCheck_iff
#axiomsClean backwardCheck_iff
#axiomsClean launchABI
end VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
