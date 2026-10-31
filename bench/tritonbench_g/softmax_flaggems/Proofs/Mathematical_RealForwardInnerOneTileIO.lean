import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerOneTileStep
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealCandidate
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealProjection

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem scatter_value {B : Nat} (s : BlockState) (Y : RegionName) (row N : Nat)
    (value : TileIndex [B] → ℝ) (i : TileIndex [B]) (hi : i.1.val<N) :
    (scatter s Y (tileOffset row N) value (tileActive N)).readMem Y (tileOffset row N i) = value i := by
  apply BlockState.scatter_readback_prop_masked_nd_of_true s (tileOffset row N) value
    (fun i ↦ tileActive N i=Bool.true) i (by simpa [tileActive] using hi)
  intro j _ he
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel he)
  · exact Subsingleton.elim _ _

theorem scatter_frame {B : Nat} (s : BlockState) (Y : RegionName) (row N : Nat)
    (value : TileIndex [B] → ℝ) :
    BlockState.WriteWithin (RealForwardInner.outputFootprint Y row N) s
      (scatter s Y (tileOffset row N) value (tileActive N)) := by
  apply BlockState.foldl_writeMem_prop_writeWithin Y (tileOffset row N) value
    (fun i ↦ tileActive N i=Bool.true) (TileShape.allIndices [B]) s
  intro i _ hi
  exact ⟨rfl,i.1.val,of_decide_eq_true hi,rfl⟩

theorem scatter_launch {B : Nat} (s : BlockState) (Y : RegionName)
    (off : TileIndex [B] → Nat) (value : TileIndex [B] → ℝ) (active : TileIndex [B] → Bool) :
    (scatter s Y off value active).pids=s.pids ∧
      (scatter s Y off value active).numPids=s.numPids ∧
      (scatter s Y off value active).undef=s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B]=lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl,rfl,rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i=Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_pids, BlockState.writeMem_numPids,
        BlockState.writeMem_undef] using ih (s.writeMem Y (off i) (value i))
    · simpa only [if_neg hi] using ih s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
