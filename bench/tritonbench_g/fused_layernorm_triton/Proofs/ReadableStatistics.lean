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

private theorem signed_address (a d r : Nat) :
    ((a : Int) + (d : Int)*(r : Int)).toNat = a+d*r := by
  rw [←Int.natCast_mul,←Int.natCast_add]
  exact Int.toNat_natCast _

private theorem signed_column (k b j d : Nat) :
    ((k : Int)*(b : Int)+(j : Int) < (d : Int)) ↔ k*b+j<d := by norm_cast

private theorem signed_column_le (k b j d : Nat) :
    ((d : Int) ≤ (k : Int)*(b : Int)+(j : Int)) ↔ d≤k*b+j := by norm_cast

private theorem signed_offset (k b j d r : Nat) :
    ((k : Int)*(b : Int)+(j : Int)+(d : Int)*(r : Int)).toNat = k*b+j+d*r := by
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

def zeroLanes (M : Algebra α) (B : Nat) : Fin B → WelfordTriple α := fun _ =>
  ⟨M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0⟩

theorem prefix_run (M : Algebra α) (equal32 : α → α → Bool) (s : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).before s = some final ∧
      WelfordLanesReady s final (zeroLanes M B) := by
  simp only [SourcePieces.statistics, SourcePieces.source, Bool.false_eq_true, ite_false,
    Kernels.kernel, Pasted.originalKernel, ComputeKernel.surfaceBody, Source.intRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, numericEq, withEquality,
    Region.cast, ofFloat, toFloat]
  constructor <;> simp [State.setReg,zeroLanes,bop,Fin.eq_zero]
  all_goals first | (constructor <;> simp [State.setReg,Fin.eq_zero]) | (funext i; simp [bop,Fin.eq_zero])

theorem first_run (M : Algebra α) (equal32 : α → α → Bool) (s t : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B)
    (h : WelfordLanesReady s t (zeroLanes M B)) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
      (t.setReg "roffset" .int [] (fun _ => 0)) = some final ∧
      WelfordLanesReady s final (firstLaneValues M s X D B) := by
  simp only [SourcePieces.statistics, SourcePieces.source, Bool.false_eq_true, ite_false,
    Kernels.kernel, Pasted.originalKernel, ComputeKernel.surfaceBody, Source.intRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, numericEq, withEquality,
    Region.cast, ofFloat, toFloat,h.mean,h.m2,h.weight,h.row,h.mem,h.indices.base,State.setReg]
  constructor <;> simp [State.setReg,firstLaneValues,zeroLanes,bop,ofFloat,toFloat,
    Fin.eq_zero,h.mem,h.pids,h.undef,h.row,h.indices.zero,h.indices.base]
  all_goals first | (constructor <;> simp [State.setReg,h.indices.zero,h.indices.base]) | rfl | (funext i; split <;> simp_all [Fin.eq_zero,signed_address,signed_column,signed_column_le,signed_offset])
  all_goals norm_num

theorem chunk_run (M : Algebra α) (equal32 : α → α → Bool) (s t : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B k : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) (hk : 0 < k) (previous : Fin B → WelfordTriple α)
    (h : WelfordLanesReady s t previous) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
      (t.setReg "roffset" .int [] (fun _ => (k*B : Nat))) = some final ∧
      WelfordLanesReady s final (advance M s X D B k previous) := by
  have hn : k*B ≠ 0 := Nat.ne_of_gt (Nat.mul_pos hk hB)
  simp only [SourcePieces.statistics, SourcePieces.source, Bool.false_eq_true, ite_false,
    Kernels.kernel, Pasted.originalKernel, ComputeKernel.surfaceBody, Source.intRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, numericEq, withEquality,
    Region.cast, ofFloat, toFloat,h.mean,h.m2,h.weight,h.row,h.mem,h.indices.base,State.setReg,hn,Nat.ne_of_gt hk,Nat.ne_of_gt hB]
  constructor <;> try simp [State.setReg,advance,reduceStep,bop,ofFloat,toFloat,
    Fin.eq_zero,h.mem,h.pids,h.undef,h.row,h.indices.zero,h.indices.base,hn,Nat.ne_of_gt hk,Nat.ne_of_gt hB]
  all_goals first | (constructor <;> simp [State.setReg,h.indices.zero,h.indices.base]) | rfl | (funext i; split <;> simp_all [Fin.eq_zero,signed_address,signed_column,signed_column_le,signed_offset])
  all_goals norm_num
  all_goals split_ifs <;> first | rfl | omega

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
    ∃ final, rangeInt (withEquality M equal32) "roffset" (k*B : Nat) D B
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body t = some final ∧
      WelfordLanesReady s final (laneRecurrence M s X D B k count previous) := by
  induction count generalizing k t previous with
  | zero =>
    have hdone : D ≤ k*B := by
      by_contra hd
      have hlt := (chunk_active D B k hB).mpr (by omega)
      omega
    refine ⟨t,?_,h⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_neg (by exact_mod_cast (not_lt.mpr hdone))]
  | succ count ih =>
    have hlt : k < blockCount D B := by omega
    have hactive := (chunk_active D B k hB).mp hlt
    obtain ⟨u,hu,hready⟩ := chunk_run M equal32 s t X W Bias Y Mean Rstd S D B k tree hB hk previous h
    obtain ⟨final,hf,hr⟩ := ih u (k+1) (by omega) (by omega) _ hready
    refine ⟨final,?_,hr⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos (by exact_mod_cast hactive),hu]
    simpa only [Nat.add_mul,Nat.one_mul,Option.bind,Nat.cast_add] using hf

theorem middle_eq (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).before =
      [.ifThen (.constBool Bool.true) (welfordTupleCode (legacyTree tree))] ++
        welfordSuffix Rstd Mean D := by
  unfold SourcePieces.normalization SourcePieces.statistics SourcePieces.source
    Kernels.kernel Pasted.originalKernel
  simp only [Bool.false_eq_true,ite_false,ComputeKernel.surfaceBody,Source.intRange?,
    Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  unfold welfordTupleCode
  rw [(expressions tree).1,(expressions tree).2.1,(expressions tree).2.2]
  rfl

theorem loop_run (M : Algebra α) (equal32 : α → α → Bool) (s p : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) (hpr : WelfordLanesReady s p (zeroLanes M B)) :
    ∃ q,rangeInt (withEquality M equal32) "roffset" 0 D B
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body p=some q ∧
      WelfordLanesReady s q (allLaneValues M s X D B) := by
  by_cases hd : D=0
  · subst D
    have hb : blockCount 0 B=0 := by
      unfold blockCount
      apply Nat.div_eq_of_lt
      omega
    refine ⟨p,by simp [rangeInt,Nat.ne_of_gt hB],?_⟩
    simpa [allLaneValues,hb,laneRecurrence,firstLaneValues,zeroLanes] using hpr
  · obtain ⟨u,hu,hur⟩ := first_run M equal32 s p X W Bias Y Mean Rstd S D B tree hpr
    obtain ⟨q,hq,hqr⟩ := chunks_run M equal32 s u X W Bias Y Mean Rstd S D B 1
      (blockCount D B-1) tree hB (by omega) rfl _ hur
    refine ⟨q,?_,hqr⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos (by omega),hu]
    simpa only [zero_add,Nat.one_mul,Option.bind] using hq

theorem statistics_ready (M : Algebra α) (equal32 : α → α → Bool) (s : State α)
    (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (tree : Kernels.ReductionTree B)
    (hB : 0 < B) :
    ∃ final, run (withEquality M equal32)
      ((SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).before ++
       [.forRangeInt "roffset"
         (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).start
         (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stop
         (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stride
         (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body] ++
       (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).before) s = some final ∧
      StatisticsReady s final Mean Rstd D B M
        (treeValue M equal32 (allLaneValues M s X D B) (legacyTree tree)) := by
  obtain ⟨p,hp,hpr⟩ := prefix_run M equal32 s X W Bias Y Mean Rstd S D B tree
  obtain ⟨q,hq,hqr⟩ := loop_run M equal32 s p X W Bias Y Mean Rstd S D B tree hB hpr
  change rangeInt (withEquality M equal32) "roffset" 0 (Int.ofNat D) B
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body p=some q at hq
  obtain ⟨u,hu,hur⟩ := welford_tuple_run M equal32 s q _ hqr (legacyTree tree)
  obtain ⟨final,hf,hready⟩ := statistics_suffix_ready M equal32 s u Mean Rstd D B _ hur
  refine ⟨final,?_,hready⟩
  rw [middle_eq]
  simp only [run_append,hp,Option.bind]
  have hstart : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).start = .ref .int [] "zero" := rfl
  have hstop : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stop = .castNatToInt (.constNat D) := rfl
  have hstride : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stride = .constNat B := rfl
  simp only [hstart,hstop,hstride,run,step,evalExpr,evalOp_unfold,Option.bind,hpr.indices.zero,hq,hu,hf,↓reduceIte,bind,pure]
  simpa only [Option.bind_some,run_append,run,step,evalOp_unfold,Option.bind,bind,pure,hu] using hf

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableStatistics
