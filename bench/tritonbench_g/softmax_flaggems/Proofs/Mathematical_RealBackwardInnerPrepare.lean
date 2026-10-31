import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def columnSum {B C : Nat} (z : TileIndex [C,B] → ℝ) (k : Fin C) : ℝ :=
  ∑ i : Fin B, z (k,i,PUnit.unit)

noncomputable def finalSum (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (k : Fin C) : ℝ :=
  columnSum (laneSum sb P G M N rbase B C (RealAddress.rounds N B)) k

theorem finalSum_eq (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (hB : 0 < B) (k : Fin C) :
    finalSum sb P G M N rbase B C k =
      ∑ n ∈ Finset.range (RealAddress.rounds N B * B),
        sample sb P M N (rbase+k.val) n * sample sb G M N (rbase+k.val) n := by
  unfold finalSum columnSum laneSum
  exact RealMath.backwardLaneAccumulator_reduce B (RealAddress.rounds N B) hB
    (fun n ↦ sample sb P M N (rbase+k.val) n *
      sample sb G M N (rbase+k.val) n)

structure ReducedInv (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase 0⟩
  sumReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (finalSum sb P G M N rbase B C i.1)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
