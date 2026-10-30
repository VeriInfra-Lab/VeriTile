import bench.tritonbench_g.fused_layernorm_triton.Proofs.SourcePieces
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeGeometry
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableStatistics
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def legacyTree {N : Nat} : Kernels.ReductionTree N → FPStages.ReductionTree N
  | .leaf i => .leaf i
  | .node l r => .node (legacyTree l) (legacyTree r)

theorem expressions {N : Nat} (t : Kernels.ReductionTree N) :
    (FPStages.treeExpressions (legacyTree t)).mean =
      (DSL.WelfordSource.expressions (.ref .real [1,N] "tmp3_mean")
        (.ref .real [1,N] "tmp3_m2") (.ref .real [1,N] "tmp3_weight") (Kernels.sourceTree t)).mean ∧
    (FPStages.treeExpressions (legacyTree t)).m2 =
      (DSL.WelfordSource.expressions (.ref .real [1,N] "tmp3_mean")
        (.ref .real [1,N] "tmp3_m2") (.ref .real [1,N] "tmp3_weight") (Kernels.sourceTree t)).m2 ∧
    (FPStages.treeExpressions (legacyTree t)).weight =
      (DSL.WelfordSource.expressions (.ref .real [1,N] "tmp3_mean")
        (.ref .real [1,N] "tmp3_m2") (.ref .real [1,N] "tmp3_weight") (Kernels.sourceTree t)).weight := by
  induction t with
  | leaf i => exact ⟨rfl,rfl,rfl⟩
  | node l r hl hr =>
    simp only [legacyTree, FPStages.treeExpressions, Kernels.sourceTree,
      DSL.WelfordSource.expressions, FPStages.combineExpressions,
      DSL.WelfordSource.combine, hl.1, hl.2.1, hl.2.2, hr.1, hr.2.1, hr.2.2]
    trivial

variable {α : Type} [Inhabited α]

theorem prefix_run (M : Algebra α) (equal32 : α → α → Bool) (s : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).before s = some final ∧
      WelfordLanesReady s final (firstLaneValues M s X D B) := by
  simp only [SourcePieces.statistics, SourcePieces.source, Bool.false_eq_true, ite_false,
    Kernels.kernel, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, numericEq, withEquality,
    Region.cast, ofFloat, toFloat]
  constructor <;> simp [State.setReg, firstLaneValues, bop, ofFloat, toFloat, Fin.eq_zero]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [bop])
  all_goals norm_num

theorem chunk_run (M : Algebra α) (equal32 : α → α → Bool) (s t : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B k : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) (hk : 0 < k) (previous : Fin B → WelfordTriple α)
    (h : WelfordLanesReady s t previous) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
      (t.setReg "roffset" .nat [] (fun _ => k*B)) = some final ∧
      WelfordLanesReady s final (advance M s X D B k previous) := by
  have hn : k*B ≠ 0 := Nat.ne_of_gt (Nat.mul_pos hk hB)
  simp only [SourcePieces.statistics, SourcePieces.source, Bool.false_eq_true, ite_false,
    Kernels.kernel, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, numericEq, withEquality,
    Region.cast, ofFloat, toFloat, h.mean, h.m2, h.weight, h.row, h.mem,
    State.setReg, hn]
  constructor <;> simp [State.setReg, advance, reduceStep, bop, ofFloat, toFloat,
    Fin.eq_zero, h.mem, h.pids, h.undef, h.row, hn]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [bop])
  all_goals norm_num

theorem chunk_active (D B k : Nat) (hB : 0 < B) :
    k < blockCount D B ↔ k*B < D := by
  unfold blockCount
  constructor
  · intro h
    have hx := (Nat.le_div_iff_mul_le hB).mp (show k+1 ≤ (D+B-1)/B by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hx
    omega
  · intro h
    have hx : (k+1)*B ≤ D+B-1 := by simp only [Nat.add_mul, Nat.one_mul]; omega
    have hl := (Nat.le_div_iff_mul_le hB).mpr hx
    omega

theorem chunks_run (M : Algebra α) (equal32 : α → α → Bool) (s t : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B k count : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) (hk : 0 < k) (hc : count = blockCount D B - k)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous) :
    ∃ final, range (withEquality M equal32) "roffset" (k*B) D B
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body t = some final ∧
      WelfordLanesReady s final (laneRecurrence M s X D B k count previous) := by
  induction count generalizing k t previous with
  | zero =>
    have hdone : D ≤ k*B := by
      by_contra hd
      have hlt := (chunk_active D B k hB).mpr (by omega)
      omega
    exact ⟨t, range_done _ _ _ _ _ _ _ hdone, h⟩
  | succ count ih =>
    have hlt : k < blockCount D B := by omega
    have hactive := (chunk_active D B k hB).mp hlt
    obtain ⟨u,hu,hready⟩ := chunk_run M equal32 s t X W Bias Y Mean Rstd S D B k tree hB hk previous h
    obtain ⟨final,hf,hr⟩ := ih u (k+1) (by omega) (by omega) _ hready
    refine ⟨final,?_,hr⟩
    rw [range_step _ _ _ _ _ _ _ (by omega) hactive, hu]
    simpa only [Nat.add_mul, Nat.one_mul, Option.bind] using hf

theorem middle_eq (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).before =
      [.ifThen (.constBool Bool.true) (welfordTupleCode (legacyTree tree))] ++
        welfordSuffix Rstd Mean D := by
  unfold SourcePieces.normalization SourcePieces.statistics SourcePieces.source
    Kernels.kernel
  simp only [Bool.false_eq_true, ite_false, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  unfold welfordTupleCode
  rw [(expressions tree).1, (expressions tree).2.1, (expressions tree).2.2]
  rfl

theorem statistics_ready (M : Algebra α) (equal32 : α → α → Bool) (s : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) :
    ∃ final, run (withEquality M equal32)
      ((SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).before ++
       [.forRange "roffset" B D B
         (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body] ++
       (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).before) s = some final ∧
      StatisticsReady s final Mean Rstd D M
        (treeValue M equal32 (allLaneValues M s X D B) (legacyTree tree)) := by
  obtain ⟨p,hp,hpr⟩ := prefix_run M equal32 s X W Bias Y Mean Rstd S D B tree
  obtain ⟨q,hq,hqr⟩ := chunks_run M equal32 s p X W Bias Y Mean Rstd S D B 1
    (blockCount D B-1) tree hB (by omega) rfl _ hpr
  obtain ⟨u,hu,hur⟩ := welford_tuple_run M equal32 s q _ hqr (legacyTree tree)
  obtain ⟨final,hf,hready⟩ := statistics_suffix_ready M equal32 s u Mean Rstd D _ hur
  refine ⟨final,?_,hready⟩
  rw [middle_eq]
  simp only [Nat.one_mul] at hq
  simp only [run_append, hp, run, step, evalExpr, evalOp_unfold, Option.bind,
    bind, pure, hq, hu, hf, ↓reduceIte]

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableStatistics
