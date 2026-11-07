import bench.tritonbench_g.apply_penalty.Proofs.Mathematical_Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Execution of the literal repair body, with signed integer input bindings.
The mathematical counts and addresses are recovered only from actual integer
loads pinned to their nonnegative values. -/
namespace VeriTile.Bench.TritonBenchG.ApplyPenalty
open VeriTile.Triton
open VeriTile.Bench.Repairs.ApplyPenalty (sourceKernel callSignature)
noncomputable section
set_option maxHeartbeats 200000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

noncomputable def penaltyValuePure (g₁ g₂ g₃ : ℝ)
    (cnts : Fin B → Nat) (xs : Fin B → ℝ) (j : Fin B) : ℝ :=
  TiledOptimizer.lionPenalty (xs j) (cnts j : ℝ) g₃ g₂ g₁

def sourceExecution (L P F R : Region .real) (I C S : Region .nat)
    (stride skip B : Nat) (s : BlockState) :
    {t : BlockState // exec ((apply_penalty L P F R I C S stride skip B).toAlgKernel) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [exec,apply_penalty,sourceKernel,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    ComputeDType.eraseDType,stepStmts,stepStmt,evalOp,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.uop,Tile.ptrAdd,Tile.select,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,ComparableDType.gt]
  rfl

private theorem maskedMemCongr {ι : Type} (lanes : List ι) (L : RegionName)
    (mask : ι → Prop) [DecidablePred mask] (a b : ι → Nat) (x y : ι → ℝ)
    (s t : BlockState) (base : s.mem=t.mem)
    (values : ∀ i,mask i → a i=b i ∧ x i=y i) :
    (lanes.foldl (fun acc i => if mask i then acc.writeMem L (a i) (x i) else acc) s).mem =
      (lanes.foldl (fun acc i => if mask i then acc.writeMem L (b i) (y i) else acc) t).mem := by
  induction lanes generalizing s t with
  | nil => exact base
  | cons head tail ih =>
    simp only [List.foldl_cons]
    by_cases h : mask head
    · simp only [h,if_true]
      obtain ⟨ha,hx⟩ := values head h
      apply ih
      · funext r o
        simp only [BlockState.writeMem,ha,hx,base]
    · simpa only [h,if_false] using ih s t base

theorem apply_penalty_region_run
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
    (ids cnts : Fin BLOCK_P → Nat) (xs : Fin BLOCK_P → ℝ)
    (hg₁ : s₀.readMem presence_penalty (s₀.pids 0) = g₁)
    (hg₂ : s₀.readMem freqency_penalty (s₀.pids 0) = g₂)
    (hg₃ : s₀.readMem repetition_penalty (s₀.pids 0) = g₃)
    (hm₁ : s₀.readMemValue .int p_cumsum_seq_len (s₀.pids 0) = (m₁ : Int))
    (hm₂ : s₀.readMemValue .int p_cumsum_seq_len (s₀.pids 0 + 1) = (m₂ : Int))
    (hid : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .int p_token_ids (m₁ + j.val) = (ids j : Int))
    (hcnt : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .int p_token_counts (m₁ + j.val) = (cnts j : Int))
    (hx : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMem Logits (s₀.pids 0 * stride_logit_b + ids j) = xs j) :
    ∃ s1, exec ((apply_penalty Logits presence_penalty freqency_penalty
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
          s1.mem r o = s₀.mem r o) := by
  let t := (sourceExecution Logits presence_penalty freqency_penalty repetition_penalty
    p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P s₀).val
  let clean := (TileShape.allIndices [BLOCK_P]).foldl
    (fun acc (i : TileIndex [BLOCK_P]) => if m₁+i.1.val<m₂ then
      acc.writeMem Logits (s₀.pids 0*stride_logit_b+ids i.1)
        (penaltyValuePure g₁ g₂ g₃ cnts xs i.1) else acc) s₀
  have memory : t.mem=clean.mem := by
    dsimp only [t,sourceExecution,clean]
    simp only [hm₁,hm₂,← Int.natCast_add,← Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
    apply maskedMemCongr
    · funext r o
      simp only [BlockState.setReg_mem]
    · intro i hi
      simp [hid i.1 hi,hcnt i.1 hi,hx i.1 hi,hg₁,hg₂,hg₃,hi,
        penaltyValuePure,TiledOptimizer.lionPenalty,Int.natCast_add,Int.natCast_mul]
      constructor
      · simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
      · by_cases pos : 0<xs i.1 <;>
          simp [pos,WithBot.some_eq_coe,WithBot.realMul,Option.map₂] <;> rfl
  refine ⟨t,(sourceExecution _ _ _ _ _ _ _ _ _ _ _).property,?_,?_⟩
  · intro inj j hj
    unfold BlockState.readMem
    rw [memory]
    change clean.readMem Logits (s₀.pids 0*stride_logit_b+ids j)=_
    exact BlockState.scatter_readback_prop_masked_list_of_true
      (TileShape.allIndices [BLOCK_P]) s₀
      (fun i : TileIndex [BLOCK_P] => s₀.pids 0*stride_logit_b+ids i.1)
      (fun i => penaltyValuePure g₁ g₂ g₃ cnts xs i.1)
      (fun i => m₁+i.1.val<m₂) (j,PUnit.unit)
      (TileShape.allIndices_nodup [BLOCK_P]) (TileShape.mem_allIndices [BLOCK_P] _) hj
      (by
        intro k _ hk same
        have eq := inj k.1 j hk hj same
        cases k with | mk head tail => cases tail; cases eq; rfl)
  · intro r o outside
    rw [memory]
    apply foldl_store_preserve_cell
    intro i _ hi same
    rcases outside with hr | miss
    · exact hr same.1.symm
    · exact miss i.1 hi same.2.symm

#axiomsClean sourceExecution
#axiomsClean apply_penalty_region_run

theorem apply_penalty_flattenOk
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    ((apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [apply_penalty, VeriTile.Bench.Repairs.ApplyPenalty.sourceKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, ComputeDType.eraseDType, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

theorem traceSafe_pinned
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s B : Nat) (bounds : RegionBounds) (s : BlockState)
    (m₁ m₂ : Nat) (ids : Fin B → Nat)
    (hm₁ : s.readMemValue .int p_cumsum_seq_len (s.pids 0) = (m₁ : Int))
    (hm₂ : s.readMemValue .int p_cumsum_seq_len (s.pids 0 + 1) = (m₂ : Int))
    (hids : ∀ j : Fin B, m₁ + j.val < m₂ →
      s.readMemValue .int p_token_ids (m₁ + j.val) = (ids j : Int))
    (hbf₁ : s.pids 0 < bounds presence_penalty)
    (hbf₂ : s.pids 0 < bounds freqency_penalty)
    (hbf₃ : s.pids 0 < bounds repetition_penalty)
    (hbm₁ : s.pids 0 < bounds p_cumsum_seq_len)
    (hbm₂ : s.pids 0 + 1 < bounds p_cumsum_seq_len)
    (hbi : ∀ j : Fin B, m₁ + j.val < m₂ → m₁ + j.val < bounds p_token_ids)
    (hbc : ∀ j : Fin B, m₁ + j.val < m₂ → m₁ + j.val < bounds p_token_counts)
    (hbr : ∀ j : Fin B, m₁ + j.val < m₂ →
      s.pids 0 * stride_logit_b + ids j < bounds Logits) :
    Kernel.TraceSafe bounds (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s B).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp [apply_penalty,sourceKernel,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    ComputeDType.eraseDType,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MaskOpt.SafeAt,MemAccess.SafeAt,stepStmt,stepStmts,evalOp.eq_def,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,tile_elementwise,
    Bool.and_eq_true,Tile.bop,Tile.cop,Tile.uop,Tile.ptrAdd,Tile.select,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
    ComparableDType.lt,ComparableDType.gt,hm₁,hm₂,
    hbf₁,hbf₂,hbf₃,hbm₁,hbm₂]
  simp only [← Int.natCast_add,← Int.natCast_mul,Int.ofNat_lt,Int.toNat_natCast]
  refine ⟨fun a ha => hbi a ha,fun a ha => hbc a ha,?_,?_⟩ <;>
    (intro a ha
     simp only [ha,if_true,hids a ha,← Int.natCast_add,← Int.natCast_mul,Int.toNat_natCast]
     exact hbr a ha)

#axiomsClean apply_penalty_flattenOk
#axiomsClean traceSafe_pinned
end
end VeriTile.Bench.TritonBenchG.ApplyPenalty
