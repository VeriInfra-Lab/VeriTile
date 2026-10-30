import bench.repairs.tritonbench_g.apply_penalty.Kernels
import bench.tritonbench_g.apply_penalty.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! The raw source keeps its signed registers. Only complete mathematical
memory observations are connected to the earlier nonnegative-index program;
there is no claim that their register states or ASTs are identical. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.Projection
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ApplyPenalty
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def current (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) : ComputeKernel :=
  if candidate then reciprocalKernel L P F R I C S stride skip B
  else kernel L P F R I C S stride skip B

def mathematical (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) : ComputeKernel :=
  if candidate then apply_penalty_reciprocal L P F R I C S stride skip B
  else apply_penalty L P F R I C S stride skip B

private theorem fold_mem {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).mem =
      items.foldl (fun mem i => if active i then
        fun r off => if r=o ∧ off=offset i then MemCell.real (value i) else mem r off
        else mem) u.mem := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi,ite_true,ih]
      rfl
    · simp only [hi,ite_false,ih]

/-- Exact Nat/Int address and mask normalization, preserving every tagged cell
and the execution-failure result for arbitrary initial states and aliases. -/
theorem algorithm (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) (s : BlockState) :
    (exec (current candidate L P F R I C S stride skip B).toAlgKernel s).map BlockState.mem =
    (exec (mathematical candidate L P F R I C S stride skip B).toAlgKernel s).map BlockState.mem := by
  cases candidate <;>
    simp [current,mathematical,kernel,reciprocalKernel,apply_penalty,apply_penalty_reciprocal,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
      exec,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,
      Tile.select,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
      ComparableDType.lt,ComparableDType.gt]
  all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
  all_goals
    simp only [fold_mem]
    congr 1
    funext mem i
    by_cases ha : s.readMemValue .nat S (s.pids 0)+i.1.val <
      s.readMemValue .nat S (s.pids 0+1)
    all_goals simp only [ha,↓reduceIte,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]


theorem mathematical_flattenOk (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) :
    (mathematical candidate L P F R I C S stride skip B).toAlgKernel.FlattenOk := by
  cases candidate <;>
    simp [mathematical,apply_penalty,apply_penalty_reciprocal,Kernel.FlattenOk,ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,StmtList.FlattenOk,Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

private theorem cast_add_ite (a b : Nat) (p : Prop) [Decidable p] :
    ((a:Int) + if p then (b:Int) else 0).toNat = a + if p then b else 0 := by
  by_cases hp : p <;> simp only [hp,ite_true,ite_false,Int.add_zero,
    ←Int.natCast_add,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero,Nat.add_zero]

private theorem cast_ite (b : Nat) (p : Prop) [Decidable p] :
    (if p then (b:Int) else 0).toNat = if p then b else 0 := by
  by_cases hp : p <;> simp [hp]

private theorem cast_neg_ite (b : Nat) (p : Prop) [Decidable p] :
    (-(if p then (b : Int) else 0)).toNat = 0 := by
  split <;> simp_all

/-- The actual signed-index loads/stores reach exactly the same addresses and
masks as the independent mathematical view, including all inactive lanes. -/
theorem traceSafe (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (current candidate L P F R I C S stride skip B).toAlgKernel s ↔
    Kernel.TraceSafe bounds (mathematical candidate L P F R I C S stride skip B).toAlgKernel s := by
  cases candidate <;>
    simp [current,mathematical,kernel,reciprocalKernel,apply_penalty,apply_penalty_reciprocal,
      Kernel.TraceSafe,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Except.bind,bind,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
      MaskOpt.SafeAt,MemAccess.SafeAt,stepStmt,stepStmts,evalOp.eq_def,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,BlockState.readMemValue,BlockState.readMemTyped,BlockState.readMem,tile_elementwise,Bool.and_eq_true,
      Tile.bop,Tile.cop,Tile.uop,Tile.ptrAdd,Tile.ptrSub,Tile.select,NumericDType.add,NumericDType.mul,
      NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.gt]
  all_goals
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt,
      cast_add_ite,cast_ite,cast_neg_ite,and_self]
  all_goals simp

theorem execution_of_mathematical (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) (s t : BlockState)
    (he : exec (mathematical candidate L P F R I C S stride skip B).toAlgKernel s = some t) :
    ∃ v, exec (current candidate L P F R I C S stride skip B).toAlgKernel s = some v ∧ v.mem=t.mem := by
  have hm := algorithm candidate L P F R I C S stride skip B s
  rw [he] at hm
  cases hx : exec (current candidate L P F R I C S stride skip B).toAlgKernel s with
  | none => simp [hx] at hm
  | some v => exact ⟨v,rfl,by simpa only [hx,Option.map_some,Option.some.injEq] using hm⟩

theorem region_run (candidate : Bool)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
    (ids cnts : Fin BLOCK_P → Nat) (xs : Fin BLOCK_P → ℝ)
    (hg₁ : s₀.readMem presence_penalty (s₀.pids 0) = g₁)
    (hg₂ : s₀.readMem freqency_penalty (s₀.pids 0) = g₂)
    (hg₃ : s₀.readMem repetition_penalty (s₀.pids 0) = g₃)
    (hm₁ : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0) = m₁)
    (hm₂ : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0 + 1) = m₂)
    (hid : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .nat p_token_ids (m₁ + j.val) = ids j)
    (hcnt : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .nat p_token_counts (m₁ + j.val) = cnts j)
    (hx : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMem Logits (s₀.pids 0 * stride_logit_b + ids j) = xs j) :
    ∃ s1, exec ((current candidate Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b stride_logit_s BLOCK_P).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin BLOCK_P, m₁ + j.val < m₂ → m₁ + k.val < m₂ →
            s₀.pids 0 * stride_logit_b + ids j
              = s₀.pids 0 * stride_logit_b + ids k → j = k) →
          ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
            s1.readMem Logits (s₀.pids 0 * stride_logit_b + ids j)
              = penaltyValuePure g₁ g₂ g₃ cnts xs j)
      ∧ (∀ r o,
          (r ≠ (Logits : RegionName) ∨ ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
            o ≠ s₀.pids 0 * stride_logit_b + ids j) →
          s1.mem r o = s₀.mem r o)  := by
  obtain ⟨t,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.ApplyPenalty.apply_penalty_region_run
    Logits presence_penalty freqency_penalty repetition_penalty p_token_ids p_token_counts
    p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P s₀ g₁ g₂ g₃ m₁ m₂ ids cnts xs
    hg₁ hg₂ hg₃ hm₁ hm₂ hid hcnt hx
  have oldExec : exec (mathematical candidate Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P).toAlgKernel s₀=some t := by
    cases candidate
    · exact he
    · exact (VeriTile.Bench.TritonBenchG.ApplyPenalty.reciprocal_exec
        Logits presence_penalty freqency_penalty repetition_penalty p_token_ids p_token_counts
        p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P s₀).trans he
  obtain ⟨v,ve,vm⟩ := execution_of_mathematical candidate Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P s₀ t oldExec
  refine ⟨v,ve,?_,?_⟩
  · intro injective j hj
    simpa only [BlockState.readMem,vm] using hv injective j hj
  · intro r o ho
    simpa only [vm] using hf r o ho

theorem safe_pinned (candidate : Bool)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s B : Nat) (bounds : RegionBounds) (s : BlockState)
    (m₁ m₂ : Nat) (ids : Fin B → Nat)
    (hm₁ : s.readMemValue .nat p_cumsum_seq_len (s.pids 0) = m₁)
    (hm₂ : s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) = m₂)
    (hids : ∀ j : Fin B, m₁ + j.val < m₂ →
      s.readMemValue .nat p_token_ids (m₁ + j.val) = ids j)
    (hbf₁ : s.pids 0 < bounds presence_penalty)
    (hbf₂ : s.pids 0 < bounds freqency_penalty)
    (hbf₃ : s.pids 0 < bounds repetition_penalty)
    (hbm₁ : s.pids 0 < bounds p_cumsum_seq_len)
    (hbm₂ : s.pids 0 + 1 < bounds p_cumsum_seq_len)
    (hbi : ∀ j : Fin B, m₁ + j.val < m₂ → m₁ + j.val < bounds p_token_ids)
    (hbc : ∀ j : Fin B, m₁ + j.val < m₂ → m₁ + j.val < bounds p_token_counts)
    (hbr : ∀ j : Fin B, m₁ + j.val < m₂ →
      s.pids 0 * stride_logit_b + ids j < bounds Logits) :
    Kernel.TraceSafe bounds (current candidate Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s B).toAlgKernel s  := by
  apply (traceSafe candidate Logits presence_penalty freqency_penalty repetition_penalty
    p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s B bounds s).mpr
  cases candidate
  · exact VeriTile.Bench.TritonBenchG.ApplyPenalty.traceSafe_pinned
      Logits presence_penalty freqency_penalty repetition_penalty p_token_ids p_token_counts
      p_cumsum_seq_len stride_logit_b stride_logit_s B bounds s m₁ m₂ ids
      hm₁ hm₂ hids hbf₁ hbf₂ hbf₃ hbm₁ hbm₂ hbi hbc hbr
  · exact VeriTile.Bench.TritonBenchG.ApplyPenalty.reciprocal_traceSafe_pinned
      Logits presence_penalty freqency_penalty repetition_penalty p_token_ids p_token_counts
      p_cumsum_seq_len stride_logit_b stride_logit_s B bounds s m₁ m₂ ids
      hm₁ hm₂ hids hbf₁ hbf₂ hbf₃ hbm₁ hbm₂ hbi hbc hbr

#axiomsClean algorithm
#axiomsClean mathematical_flattenOk
#axiomsClean region_run
#axiomsClean safe_pinned
#axiomsClean traceSafe
#axiomsClean execution_of_mathematical
end
end VeriTile.Bench.Repairs.ApplyPenalty.Projection
