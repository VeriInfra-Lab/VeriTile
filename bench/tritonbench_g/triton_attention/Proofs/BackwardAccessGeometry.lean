import bench.tritonbench_g.triton_attention.Proofs.AccessGeometry
import bench.tritonbench_g.triton_attention.Proofs.PreprocessAccessSafety

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]

theorem backward_head (c : Shape) (scale : ℝ) (s : State α) :
    backwardHeadRow (backwardParameters c scale) s=s.pids 0*c.rows := by
  simp only [backwardHeadRow,backwardParameters,Nat.mul_div_cancel _ c.dim_pos]
  calc
    s.pids 0 / c.heads * (c.heads*c.rows) + s.pids 0 % c.heads * c.rows =
        (s.pids 0/c.heads*c.heads+s.pids 0%c.heads)*c.rows := by ring
    _ = _ := by rw [Nat.div_add_mod']

theorem backward_element_bound (c : Shape) (scale : ℝ) (s : State α)
    (legal : s.pids 0<c.groups) (block : Nat) (live : block<c.blocks)
    (i : BLane (backwardParameters c scale)) :
    backwardOffset (backwardParameters c scale) s block i<c.elements := by
  rw [backwardOffset,backward_head]
  simpa only [backwardParameters,Nat.add_assoc] using
    element_bound c _ _ _ legal (tile_row_bound c _ _ live i.1.isLt) i.2.1.isLt

theorem backward_row_bound (c : Shape) (scale : ℝ) (s : State α)
    (legal : s.pids 0<c.groups) (block : Nat) (live : block<c.blocks)
    (i : BRow (backwardParameters c scale)) :
    backwardRowOffset (backwardParameters c scale) s block i<c.groups*c.rows := by
  have address : backwardRowOffset (backwardParameters c scale) s block i=
      s.pids 0*c.rows+(block*128+i.1.val) := by
    simp [backwardRowOffset,backwardParameters,Shape.rows,Nat.mul_comm,Nat.add_assoc]
  rw [address]
  exact row_bound c _ _ legal (tile_row_bound c _ _ live i.1.isLt)

theorem preprocess_row_bound (c : Shape) (s : State α)
    (legal : s.pids 0<c.blocks*c.groups) (i : PreRow (preParameters c)) :
    preRow (preParameters c) s i<c.groups*c.rows := by
  have hi : i.1.val<128 := i.1.isLt
  have size : c.groups*c.rows=(c.blocks*c.groups)*128 := by unfold Shape.rows;ring
  change s.pids 0*128+i.1.val<_
  rw [size]
  omega

theorem preprocess_element_bound (c : Shape) (s : State α)
    (legal : s.pids 0<c.blocks*c.groups) (i : PreLane (preParameters c)) :
    preOffset (preParameters c) s i<c.elements := by
  have row := preprocess_row_bound c s legal (i.1,PUnit.unit)
  have col : i.2.1.val<c.dim := i.2.1.isLt
  change s.pids 0*128+i.1.val<c.groups*c.rows at row
  change (s.pids 0*128+i.1.val)*c.dim+i.2.1.val<c.groups*c.rows*c.dim
  nlinarith

theorem preprocess_kernel_safe (rest : Algebra α) (P : Primitives α)
    (c : Shape) (s : State α) (legal : s.pids 0<c.blocks*c.groups)
    (bounds : RegionName→Nat) (backed : Bounds c bounds) :
    TraceSafe (backend rest P) bounds (preprocessKernel c) s := by
  apply preprocess_safe rest P (preParameters c) s bounds
  · exact fun i=>lt_of_lt_of_le (preprocess_element_bound c s legal i) (backed.1 "Out" (by simp))
  · exact fun i=>lt_of_lt_of_le (preprocess_element_bound c s legal i) (backed.1 "DO" (by simp))
  · exact fun i=>lt_of_lt_of_le (preprocess_row_bound c s legal i) (backed.2 "L" (by simp))
  · exact fun i=>lt_of_lt_of_le (preprocess_element_bound c s legal i) (backed.1 "NewDO" (by simp))
  · exact fun i=>lt_of_lt_of_le (preprocess_row_bound c s legal i) (backed.2 "Delta" (by simp))

#axiomsClean backward_element_bound
#axiomsClean preprocess_kernel_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
