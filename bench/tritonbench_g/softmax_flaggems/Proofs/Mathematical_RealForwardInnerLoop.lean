import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerFullStep
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardModel
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

noncomputable def laneState (sb : BlockState) (X : RegionName)
    (row N B c : Nat) (i : TileIndex [B]) : WithBot ℝ × ℝ :=
  RealForwardModel.state N B i.1.val c (fun n ↦ sb.readMem X (row * N + n))

theorem laneState_zero (sb : BlockState) (X : RegionName)
    (row N B : Nat) (i : TileIndex [B]) :
    laneState sb X row N B 0 i = (⊥, 0) := by
  simp [laneState, RealForwardModel.state, RealForwardModel.values]

theorem laneState_succ (sb : BlockState) (X : RegionName)
    (row N B c : Nat) (i : TileIndex [B]) :
    laneState sb X row N B (c + 1) i =
      RealMath.maskedOnlineStep (laneState sb X row N B c i)
        (score sb X row N (c * B) B i) := by
  simpa [laneState, score] using RealForwardModel.state_succ N B i.1.val c
    (fun n ↦ sb.readMem X (row * N + n))

structure Pass1Inv (sb : BlockState) (X Y : RegionName)
    (row N B c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  inputReg : st.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N))
  outputReg : st.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N))
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [B] "m" =
    some ⟨fun i ↦ (laneState sb X row N B c i).1⟩
  sumReg : st.regs .real [B] "z" =
    some ⟨fun i ↦ some (laneState sb X row N B c i).2⟩

/-- Exact aligned source loop, including a zero-iteration prefix. -/
theorem range_aligned {idx : RegName} {q B : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} (hB : 0 < B)
    (hstep : ∀ c s, c < q → P c s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar (c * B))) = some s' ∧ P (c + 1) s') :
    ∀ c s, c ≤ q → P c s →
      ∃ s', stepForRangeAux idx (c * B) (q * B) B body s = some s' ∧ P q s' := by
  intro c s
  induction h : q - c using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hc hp
    by_cases he : c = q
    · subst c
      exact ⟨s, stepForRangeAux.step_ge (Nat.ne_of_gt hB) (Nat.le_refl _), hp⟩
    · have hlt : c < q := by omega
      obtain ⟨next, hexec, hn⟩ := hstep c s hlt hp
      obtain ⟨last, hr, hl⟩ := ih (q - (c + 1)) (by omega) (c + 1) next rfl (by omega) hn
      refine ⟨last, ?_, hl⟩
      rw [stepForRangeAux.step_lt (Nat.ne_of_gt hB) (Nat.mul_lt_mul_of_pos_right hlt hB), hexec]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul] using hr

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
