import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealFrame
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSourceScore

/-! Draft for root validation. The first lemma observes the final scatter of
the actual source witness, using tile operators only as a semantic bridge.
The public readback below then replaces that bridge by the independently
defined scalar supremum/sum formula. All datatype erasure is explicit.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

private theorem causal_bottom_padding {α : Type} (lane K row : Nat) (x z : α) :
    (if row < lane then z else if lane < K ∧ lane ≤ row then x else z) =
      (if lane < K ∧ lane ≤ row then x else z) := by
  by_cases h : lane ≤ row
  · simp [h, Nat.not_lt.mpr h]
  · simp [h, Nat.lt_of_not_ge h]

private theorem causal_active_padding {α : Type} (lane row : Nat) (x z : α) :
    (if row < lane then z else if lane ≤ row then x else z) =
      (if lane ≤ row then x else z) := by
  by_cases h : lane ≤ row
  · simp [h, Nat.not_lt.mpr h]
  · simp [h, Nat.lt_of_not_ge h]

private theorem add_masked (p : Prop) [Decidable p] (x m : ℝ) :
    WithBot.realAdd (if p then some x else none) (if p then some m else none) =
      (if p then some (x + m) else none) := by
  by_cases h : p <;> simp [h, WithBot.realAdd]

/-- Rewrite the complete row function before tile operators duplicate it.
The checked scalar source-score theorem removes the redundant causal where;
the original masked-load and masked-add expression is otherwise retained. -/
private theorem loadedScore_function {D : Nat} (K row : Nat) (causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) :
    loadedScore K row causal mode xs masks =
      (fun i =>
        let x : WithBot ℝ :=
          if readActive K row causal i.val then some (xs i) else none
        if mode.present then
          WithBot.realAdd x
            (if readActive K row causal i.val then some (masks i) else none)
        else x) := by
  rw [loadedScore_eq_score]
  cases mode <;> funext i
  all_goals
    by_cases hi : readActive K row causal i.val = Bool.true
    · simp [score, hi, MaskMode.present, WithBot.realAdd]
    · simp [score, hi, MaskMode.present, WithBot.realAdd]

/-- Observe the exact result tile stored by the source. This statement is
not the independent scalar specification; the next theorem supplies that
bridge without rerunning or replacing the source program. -/
theorem fp32_source_tile_readback (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (s t : BlockState)
    (hExec : exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
      s = some t) (i : Fin D) :
    t.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val) =
      if i.val < K then
        WithBot.unbotD 0
          ((resultTile log
            (loadedScore K (s.pids 1) causal mode
              (fun j => s.readMem X (s.pids 0 * xm + s.pids 1 * xn + j.val))
              (fun j => s.readMem M
                (maskAddress mode (s.pids 0) (s.pids 1) sm j.val)))).data
            (i, PUnit.unit))
      else s.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val) := by
  have ht := Option.some.inj
    ((fp32_source_runState Y X M ym yn xm xn sm K D hD log mode causal fp16 s).property.symm.trans
      hExec)
  subst t
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [fp32_source_runState]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < K
    · rw [loadedScore_function]
      simp (config := { contextual := Bool.true })
        [hi, resultTile, loadedScore, readActive, MaskMode.present, maskAddress,
          shiftedTile, numeratorTile, denominatorTile, valuesTile, maximumTile,
          rowMaximum, tile_elementwise, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          Finset.sup'_eq_sup, causal_bottom_padding, causal_active_padding, add_masked]
      all_goals try simp only [causal_bottom_padding, causal_active_padding]
      all_goals rfl
    · simp [hi]

/-- Full original-source fp32 readback after mathematical datatype erasure,
for every admitted mask mode, all flags and every program's padded row.
The real-memory observer demotes bottom; extended typed values are separate. -/
theorem fp32_source_correct (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (s t : BlockState)
    (hExec : exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
      s = some t) (i : Fin D) :
    t.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val) =
      if i.val < K then
        formula K (s.pids 1) log causal mode
          (fun j => s.readMem X (s.pids 0 * xm + s.pids 1 * xn + j.val))
          (fun j => s.readMem M (maskAddress mode (s.pids 0) (s.pids 1) sm j.val)) i
      else s.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val) := by
  rw [fp32_source_tile_readback Y X M ym yn xm xn sm K D hD log mode causal fp16 s t hExec i]
  exact congrArg
    (fun value : WithBot ℝ => if i.val < K then WithBot.unbotD 0 value
      else s.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val))
    (loadedScore_result_formula K (s.pids 1) D log causal mode
      (fun j : Fin D => s.readMem X (s.pids 0 * xm + s.pids 1 * xn + j.val))
      (fun j : Fin D => s.readMem M
        (maskAddress mode (s.pids 0) (s.pids 1) sm j.val)) i)

/-- Real region IO and complete unwritten-memory frame. An absent mask
imposes no read obligation on M, and the output mask remains k < K. -/
theorem fp32_source_region_run (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool)
    (s : BlockState) (xs masks : Fin D → ℝ)
    (hx : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem X (s.pids 0 * xm + s.pids 1 * xn + j.val) = xs j)
    (hm : ∀ j, mode.present = Bool.true →
      readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem M (maskAddress mode (s.pids 0) (s.pids 1) sm j.val) = masks j) :
    ∃ t, exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
        s = some t ∧
      (∀ i : Fin D, i.val < K →
        t.readMem Y (s.pids 0 * ym + s.pids 1 * yn + i.val) =
          formula K (s.pids 1) log causal mode xs masks i) ∧
      (∀ r offset,
        (r ≠ Y ∨ ∀ i : Fin D, i.val < K →
          offset ≠ s.pids 0 * ym + s.pids 1 * yn + i.val) →
        t.mem r offset = s.mem r offset) := by
  obtain ⟨t, hrun⟩ :=
    fp32_source_exec_isSome Y X M ym yn xm xn sm K D hD log mode causal fp16 s
  refine ⟨t, hrun, ?_, ?_⟩
  · intro i hi
    have hv := fp32_source_correct Y X M ym yn xm xn sm K D hD log mode causal fp16 s t hrun i
    rw [if_pos hi] at hv
    exact hv.trans (formula_congr K (s.pids 1) log causal mode _ xs _ masks hx hm i)
  · intro r offset hmiss
    apply fp32_source_frame Y X M ym yn xm xn sm K D hD log mode causal fp16 s t hrun r offset
    intro i hi ⟨hr, hoff⟩
    rcases hmiss with hregion | hlane
    · exact hregion hr.symm
    · exact hlane i hi hoff.symm

#axiomsClean fp32_source_tile_readback
#axiomsClean fp32_source_correct
#axiomsClean fp32_source_region_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
