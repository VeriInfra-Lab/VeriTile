import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInner
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def laneSum (sb : BlockState) (P G : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) : ℝ :=
  RealMath.backwardLaneAccumulator B c (fun n ↦
    sample sb P row N K n (kbase + i.2.1.val) *
      sample sb G row N K n (kbase + i.2.1.val)) i.1

theorem laneSum_zero (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (i : TileIndex [B, C]) :
    laneSum sb P G row N K kbase B C 0 i = 0 := by
  simp [laneSum, RealMath.backwardLaneAccumulator]

theorem laneSum_succ (sb : BlockState) (P G : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) :
    laneSum sb P G row N K kbase B C (c+1) i =
      laneSum sb P G row N K kbase B C c i +
        tileSample sb P row N K kbase (c*B) i *
        tileSample sb G row N K kbase (c*B) i := by
  simp [laneSum, RealMath.backwardLaneAccumulator_eq, Finset.sum_range_succ, tileSample]

structure Pass1Inv (sb : BlockState) (P G : RegionName)
    (row N K kbase B C c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase + i.1.val⟩
  nReg : st.regs .nat [B] "offsets_n" = some ⟨fun i ↦ c*B + i.1.val⟩
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase (c*B)⟩
  sumReg : st.regs .real [B,C] "scale" =
    some ⟨fun i ↦ some (laneSum sb P G row N K kbase B C c i)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
