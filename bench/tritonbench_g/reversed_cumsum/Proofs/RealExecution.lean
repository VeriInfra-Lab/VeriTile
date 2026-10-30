import bench.tritonbench_g.reversed_cumsum.Proofs.RealBody
import bench.tritonbench_g.reversed_cumsum.Proofs.RealInit
import bench.tritonbench_g.reversed_cumsum.Proofs.SignedRange
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.SignedIndex

/-! Complete source execution, including the empty time dimension. Original
and candidate initialization are each executed, then every actual reverse
iteration establishes the same independent padded suffix and tagged frame. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution
open VeriTile.Triton RealProgram RealStep RealLoop
open VeriTile.Bench.TritonBenchG.ReversedCumsumScalar
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem mirror_invariant (initial : BlockState) (p : Layout) (l : Launch)
    (s o : RegionName) (a : Nat → ℝ) (c : Nat) (u : BlockState)
    (h : Holds initial p l s o a c u) (j : Int) :
    Holds initial p l s o a c (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  refine ⟨h.undef, h.count_le, ?_, ?_, ?_, ?_, h.input, h.output, h.frame⟩
  · simpa using h.feature
  · simpa using h.head
  · simpa using h.triangular
  · simpa using h.accumulated

theorem mirror_assign (j : Int) (u : BlockState) :
    stepStmt (.assign .int [] "i_t"
      (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")))
      (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) =
    some ((u.setReg "__rev_i_t" .int [] (Tile.scalar j)).setReg "i_t" .int [] (Tile.scalar (-j))) := by
  simp only [stepStmt, evalOp, BlockState.setReg_same, Option.bind_some,
    Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply congrArg (fun tile : Tile .int [] =>
    (u.setReg "__rev_i_t" .int [] (Tile.scalar j)).setReg "i_t" .int [] tile)
  apply Tile.ext
  intro i
  change 0 - j = -j
  omega

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p l)
    (j : Int) (u : BlockState) (hj : j < 1)
    (h : Holds initial p l s o a (SignedIndex.remaining j) u) :
    ∃ v, stepStmts (iterationBody p s o)
        (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) = some v ∧
      Holds initial p l s o a (SignedIndex.remaining (j + 1)) v := by
  have hc : (-j).toNat < numChunks p := by
    have hn := h.count_le
    rw [SignedIndex.current_chunk j hj] at hn
    omega
  have hm := mirror_invariant initial p l s o a (SignedIndex.remaining j) u h j
  rw [SignedIndex.current_chunk j hj] at hm
  obtain ⟨v, he, hv⟩ := RealLoop.iteration initial p l s o a hSO hBT hinj
    (-j).toNat (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) hc hm
  refine ⟨v, ?_, ?_⟩
  · rw [iterationBody, stepStmts.cons_some (mirror_assign j u)]
    simpa only [SignedIndex.int_index_of_nat j hj] using he
  · simpa only [SignedIndex.after_step] using hv

theorem start_eval (p : Layout) (u : BlockState) :
    evalOp (reverseStart p) u = some (Tile.scalar (1 - (numChunks p : Int))) := by
  simp only [reverseStart, stopOp, evalOp, Option.bind_some,
    Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  intro i
  change 0 - ((((p.T + p.BT - 1) / p.BT : Nat) : Int) - 1) = 1 - (numChunks p : Int)
  unfold numChunks
  omega

theorem statement_eval (p : Layout) (s o : RegionName) (u : BlockState) :
    stepStmt (statement p s o) u =
      stepForRangeIntAux "__rev_i_t" (1 - (numChunks p : Int)) 1 1
        (iterationBody p s o) u := by
  rw [statement, stepForRangeIntAux.forRangeInt_unfold, start_eval]
  simp only [evalOp, Option.bind_some, Option.bind_eq_bind, Tile.scalar]

theorem loop_executes (initial : BlockState) (p : Layout) (l : Launch) (s o : RegionName)
    (a : Nat → ℝ) (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p l)
    (u : BlockState) (h : Holds initial p l s o a (numChunks p) u) :
    ∃ v, stepStmt (statement p s o) u = some v ∧ Holds initial p l s o a 0 v := by
  obtain ⟨final, v, he, hf, hv⟩ := SignedRange.executes "__rev_i_t" 1 1
    (iterationBody p s o) (fun j v => Holds initial p l s o a (SignedIndex.remaining j) v)
    (by decide)
    (fun j v hj hv => iteration initial p l s o a hSO hBT hinj j v hj hv)
    (1 - (numChunks p : Int)) u (by simpa only [SignedIndex.at_start] using h)
  refine ⟨v, (statement_eval p s o u).trans he, ?_⟩
  simpa only [SignedIndex.at_or_after_end final hf] using hv

def Post (p : Layout) (s o : RegionName) (initial final : BlockState) : Prop :=
  (∀ (t : Nat) (j : Fin p.BS), active p (RealInit.launch initial) t j.val →
    final.mem o (address p (RealInit.launch initial) t j.val) =
      MemCell.real (expected p (RealInit.launch initial) (initial.readMem s) (initial.undef s) t j)) ∧
  BlockState.WriteWithin (footprint p (RealInit.launch initial) o) initial final

theorem source_correct (swapped : Bool) (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (initial : BlockState)
    (hSO : s ≠ o) (hBT : 0 < p.BT) (hinj : NoCollision p (RealInit.launch initial)) :
    ∃ final, exec (RealSource.source swapped dtype out_dtype s o
      p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel initial =
        some final ∧ Post p s o initial final := by
  obtain ⟨final, he, h⟩ := loop_executes initial p (RealInit.launch initial) s o
    (initial.readMem s) hSO hBT hinj (RealInit.initial p initial)
    (RealInit.invariant p initial s o (initial.readMem s) (fun _ => rfl) hBT)
  refine ⟨final, ?_, ?_, h.frame⟩
  · rw [exec, source_body, body, ProgramRewrite.append, RealInit.executes]
    change stepStmts [statement p s o] (RealInit.initial p initial) = some final
    rw [stepStmts.cons_some he, stepStmts.nil]
  · intro t j ha
    exact h.output t j ha (by simp)

#axiomsClean iteration
#axiomsClean loop_executes
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution
