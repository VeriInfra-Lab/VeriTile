import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInner
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardModel
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

noncomputable def laneState (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) : WithBot ℝ × ℝ :=
  if kbase + i.2.1.val < K then
    RealForwardModel.state N B i.1.val c
      (fun n ↦ sb.readMem X (row * N * K + n * K + (kbase + i.2.1.val)))
  else (⊥, 0)

theorem laneState_zero (sb : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (i : TileIndex [B, C]) :
    laneState sb X row N K kbase B C 0 i = (⊥, 0) := by
  simp [laneState, RealForwardModel.state, RealForwardModel.values]

theorem laneState_succ (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) :
    laneState sb X row N K kbase B C (c + 1) i =
      RealMath.maskedOnlineStep (laneState sb X row N K kbase B C c i)
        (score sb X row N K kbase (c * B) B C i) := by
  by_cases hk : kbase + i.2.1.val < K
  · simpa [laneState, score, hk] using
      RealForwardModel.state_succ N B i.1.val c
        (fun n ↦ sb.readMem X (row * N * K + n * K + (kbase + i.2.1.val)))
  · simp only [laneState, if_neg hk, score, hk, and_false, if_false]
    exact (RealMath.maskedOnlineStep_none ⊥ 0 (fun _ ↦ rfl)).symm

structure Pass1Inv (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩
  maxReg : st.regs .real [B, C] "m" =
    some ⟨fun i ↦ (laneState sb X row N K kbase B C c i).1⟩
  sumReg : st.regs .real [B, C] "z" =
    some ⟨fun i ↦ some (laneState sb X row N K kbase B C c i).2⟩

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
