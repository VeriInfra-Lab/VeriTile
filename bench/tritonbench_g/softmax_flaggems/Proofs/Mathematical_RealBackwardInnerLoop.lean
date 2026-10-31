import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInner
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def laneSum (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (i : TileIndex [C,B]) : ℝ :=
  RealMath.backwardLaneAccumulator B c (fun n ↦
    sample sb P M N (rbase+i.1.val) n *
      sample sb G M N (rbase+i.1.val) n) i.2.1

theorem laneSum_zero (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (i : TileIndex [C,B]) :
    laneSum sb P G M N rbase B C 0 i = 0 := by
  simp [laneSum, RealMath.backwardLaneAccumulator]

theorem laneSum_succ (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (i : TileIndex [C,B]) :
    laneSum sb P G M N rbase B C (c+1) i =
      laneSum sb P G M N rbase B C c i +
        tileSample sb P M N rbase (c*B) i *
        tileSample sb G M N rbase (c*B) i := by
  simp [laneSum, RealMath.backwardLaneAccumulator_eq, Finset.sum_range_succ, tileSample]

structure Pass1Inv (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase + i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ c*B + i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase (c*B)⟩
  sumReg : st.regs .real [C,B] "scale" =
    some ⟨fun i ↦ some (laneSum sb P G M N rbase B C c i)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
