import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyInterPrelude
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyDiagonal
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyOutput

/-! Address capacities follow from the existing launch grids and finite buffer
extents. The arbitrary incoming typed memory and undefined value are retained. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open FPStages (Stage diagonalLayout mergeLayout pid_bound)
open NativeLaunch (start Bounds inputExtents)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000

private theorem input_bound (stage : Stage) (bounds : RegionName → Nat)
    (hb : Bounds stage bounds) (region : RegionName) (size : Nat)
    (mem : (region,size) ∈ inputExtents stage) : size ≤ bounds region := hb.1 _ mem

theorem inter_capacity (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (Stage.inter c scale).grid) (bounds : RegionName → Nat)
    (hb : Bounds (.inter c scale) bounds) (s : State α) :
    InterCapacity c (start (.inter c scale) idx s) bounds := by
  constructor
  · simpa [start,Stage.grid,ProducerGeometry.grid,Grid.dim] using
      pid_bound (ProducerGeometry.grid c) idx 2 (by simp [ProducerGeometry.grid,Grid.rank])
  · exact input_bound _ _ hb "q" (c.heads*c.T*c.K) (by simp [inputExtents])
  · exact input_bound _ _ hb "k" (c.heads*c.T*c.K) (by simp [inputExtents])
  · exact input_bound _ _ hb "g" (c.heads*c.T*c.K) (by simp [inputExtents])
  · exact hb.2

theorem intra_capacity (c : ProducerGeometry.Shape) (BK : Nat) (scale : ℝ)
    (idx : GridIndex (Stage.intra c BK scale).grid) (bounds : RegionName → Nat)
    (hb : Bounds (.intra c BK scale) bounds) (s : State α) :
    DiagonalCapacity Bool.false (diagonalLayout c BK) (start (.intra c BK scale) idx s) bounds := by
  have hh : (start (.intra c BK scale) idx s).pids 2 < c.heads := by
    simpa [start,Stage.grid,DiagonalGeometry.grid,Grid.dim] using
      pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
  have hq := input_bound _ _ hb "q" (c.heads*c.T*c.K) (by simp [inputExtents])
  have hk := input_bound _ _ hb "k" (c.heads*c.T*c.K) (by simp [inputExtents])
  have hg := input_bound _ _ hb "g" (c.heads*c.T*c.K) (by simp [inputExtents])
  constructor
  · intro i hr he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_query _ _ _ c.heads i hh hr he) hq
  · intro i hr he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_query _ _ _ c.heads i hh hr he) hg
  · intro j hj i he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_key _ _ _ c.heads j i.val hh hj he) hk
  · intro j hj i he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_key _ _ _ c.heads j i.val hh hj he) hg
  · intro j hj i hi
    have hc : idx.toPids 1 < c.NC := by
      simpa [DiagonalGeometry.grid,Grid.dim] using
        pid_bound (DiagonalGeometry.grid c) idx 1 (by simp [DiagonalGeometry.grid,Grid.rank])
    have hj' : j < 16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
    have hcol : idx.toPids 1*16+j < c.BT := by
      rw [c.bt_eq]
      exact Geometry.packed_lt hc hj'
    have backed := Geometry.attention_bound hh hi hcol
    apply lt_of_lt_of_le _ hb.2
    simpa only [Stage.size,FPDiagonal.outputAddress,FPDiagonal.outputBase,FPDiagonal.block,diagonalLayout,
      start,Bool.false_eq_true,if_false,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]
      using backed

theorem split_capacity (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (Stage.split c scale).grid) (bounds : RegionName → Nat)
    (hb : Bounds (.split c scale) bounds) (s : State α) :
    DiagonalCapacity Bool.true (diagonalLayout c c.BK) (start (.split c scale) idx s) bounds := by
  have hh : (start (.split c scale) idx s).pids 2 < c.heads := by
    simpa [start,Stage.grid,SplitGeometry.grid,Grid.dim] using
      pid_bound (SplitGeometry.grid c) idx 2 (by simp [SplitGeometry.grid,Grid.rank])
  have hq := input_bound _ _ hb "q" (c.heads*c.T*c.K) (by simp [inputExtents])
  have hk := input_bound _ _ hb "k" (c.heads*c.T*c.K) (by simp [inputExtents])
  have hg := input_bound _ _ hb "g" (c.heads*c.T*c.K) (by simp [inputExtents])
  constructor
  · intro i hr he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_query _ _ _ c.heads i hh hr he) hq
  · intro i hr he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_query _ _ _ c.heads i hh hr he) hg
  · intro j hj i he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_key _ _ _ c.heads j i.val hh hj he) hk
  · intro j hj i he
    exact lt_of_lt_of_le (FPReadBounds.diagonal_key _ _ _ c.heads j i.val hh hj he) hg
  · intro j hj i hi
    have hf : idx.toPids 0 < Geometry.ceilDiv c.K c.BK := by
      simpa [SplitGeometry.grid,Grid.dim] using
        pid_bound (SplitGeometry.grid c) idx 0 (by simp [SplitGeometry.grid,Grid.rank])
    have hj' : j < 16 := lt_of_lt_of_le hj (Nat.min_le_left _ _)
    have backed := Geometry.scratch_bound hf hh hi hj'
    apply lt_of_lt_of_le _ hb.2
    simpa only [Stage.size,FPDiagonal.outputAddress,FPDiagonal.outputBase,diagonalLayout,start,Stage.grid,
      if_true,Geometry.scratchAddress,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,
      show (SplitGeometry.grid c).toNumPids 2=c.heads by
        simp [Grid.toNumPids,Grid.dim,Grid.rank,SplitGeometry.grid],Nat.add_comm] using backed

theorem merge_capacity (c : ProducerGeometry.Shape)
    (idx : GridIndex (Stage.merge c).grid) (bounds : RegionName → Nat)
    (hb : Bounds (.merge c) bounds) (s : State α) :
    MergeCapacity (mergeLayout c) (start (.merge c) idx s) bounds := by
  have hh : (start (.merge c) idx s).pids 2 < c.heads := by
    simpa [start,Stage.grid,DiagonalGeometry.grid,Grid.dim] using
      pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
  constructor
  · intro f hf i hr
    have hj : i.2.1.val < 16 := i.2.1.isLt
    have backed := Geometry.scratch_bound hf hh hr hj
    apply lt_of_lt_of_le _ (input_bound _ _ hb "scratch" (Geometry.ceilDiv c.K c.BK*c.heads*c.T*16) (by simp [inputExtents]))
    simpa only [FPMerge.inputAddress,mergeLayout,start,Stage.grid,Geometry.scratchAddress,
      Nat.add_mul,Nat.mul_assoc,
      show (DiagonalGeometry.grid c).toNumPids 2=c.heads by
        simp [Grid.toNumPids,Grid.dim,Grid.rank,DiagonalGeometry.grid],Nat.add_assoc,Nat.add_comm]
      using backed
  · intro i hi
    apply lt_of_lt_of_le _ hb.2
    simpa only [Stage.size,FPMerge.outputAddress,mergeLayout,start,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
      using Geometry.attention_bound hh hi.1 hi.2

theorem output_capacity (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (Stage.output c scale).grid) (bounds : RegionName → Nat)
    (hb : Bounds (.output c scale) bounds) (s : State α) :
    OutputCapacity c (start (.output c scale) idx s) bounds := by
  constructor
  · simpa [start,Stage.grid,OutputGrid.grid,Grid.dim] using
      pid_bound (OutputGrid.grid c) idx 2 (by simp [OutputGrid.grid,Grid.rank])
  · simpa [start,Stage.grid,OutputGrid.grid,Grid.dim] using
      pid_bound (OutputGrid.grid c) idx 1 (by simp [OutputGrid.grid,Grid.rank])
  · exact input_bound _ _ hb "q" (c.heads*c.T*c.K) (by simp [inputExtents])
  · exact input_bound _ _ hb "g" (c.heads*c.T*c.K) (by simp [inputExtents])
  · exact input_bound _ _ hb "h" _ (by simp [inputExtents])
  · exact input_bound _ _ hb "v" _ (by simp [inputExtents])
  · exact input_bound _ _ hb "attention" _ (by simp [inputExtents])
  · exact hb.2

#axiomsClean inter_capacity
#axiomsClean intra_capacity
#axiomsClean split_capacity
#axiomsClean merge_capacity
#axiomsClean output_capacity
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
