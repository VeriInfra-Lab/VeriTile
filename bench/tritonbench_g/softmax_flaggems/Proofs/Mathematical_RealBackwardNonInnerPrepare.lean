import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def columnSum {B C : Nat} (z : TileIndex [B,C] → ℝ) (k : Fin C) : ℝ :=
  ∑ i : Fin B, z (i,k,PUnit.unit)

noncomputable def finalSum (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (k : Fin C) : ℝ :=
  columnSum (laneSum sb P G row N K kbase B C (RealAddress.rounds N B)) k

theorem finalSum_eq (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) (k : Fin C) :
    finalSum sb P G row N K kbase B C k =
      ∑ n ∈ Finset.range (RealAddress.rounds N B * B),
        sample sb P row N K n (kbase+k.val) * sample sb G row N K n (kbase+k.val) := by
  unfold finalSum columnSum laneSum
  exact RealMath.backwardLaneAccumulator_reduce B (RealAddress.rounds N B) hB
    (fun n ↦ sample sb P row N K n (kbase+k.val) *
      sample sb G row N K n (kbase+k.val))

structure ReducedInv (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩
  nReg : st.regs .nat [B] "offsets_n" = some ⟨fun i ↦ i.1.val⟩
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase 0⟩
  sumReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (finalSum sb P G row N K kbase B C i.1)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
