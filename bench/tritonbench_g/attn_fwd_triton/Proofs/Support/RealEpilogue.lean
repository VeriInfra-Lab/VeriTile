import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Source
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealAccess
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealKeys
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealProbabilities
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealRegisterSafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealTiles
import bench.tritonbench_g.attention_score.Proofs.Mathematical_RealFrame
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealPointers
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealSource
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealEpilogue
open VeriTile.Triton RealLoads RealPointers
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def active (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD]) : Bool :=
  decide (c.startM*BM+i.1.val < c.context ∧ i.2.1.val < c.activeHead)

def address (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD]) : Nat :=
  c.base+(c.startM*BM+i.1.val)*c.strideM+i.2.1.val*c.strideD

def writeOutput (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) : BlockState :=
  (TileShape.allIndices [BM,BD]).foldl (fun st i =>
    if active c BM BD i then st.writeMem Out (address c BM BD i) (values i) else st) s

theorem normalize_executes (s : BlockState) (BM BD : Nat) (candidate : Bool)
    (a : TileIndex [BM,BD] → ℝ) (l : Fin BM → ℝ)
    (ha : s.regs .real [BM,BD] "acc" = some (finite a))
    (hl : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => l i.1))) :
    stepStmt (RealSource.normalization BM BD candidate) s =
      some (s.setReg "acc" .real [BM,BD] (finite (fun i : TileIndex [BM,BD] => a i/l i.1))) := by
  apply stepStmt_assign_eq_some
  cases candidate
  all_goals
    simp only [RealSource.normalization, Bool.false_eq_true, Bool.true_eq, ↓reduceIte,
      evalOp.eq_def, ha, hl, bind, Option.bind]
  all_goals congr 1
  all_goals
    apply Tile.ext
    rintro ⟨r,d,⟨⟩⟩
    norm_num [finite, Tile.bop_data, Tile.expandDim_data, Tile.scalar,
      TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.div,
      WithBot.realMul, WithBot.realDiv, div_eq_mul_inv, -Tile.scalar_eta]

theorem frame (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) (r : RegionName) (a : Nat)
    (h : r ≠ Out ∨ ∀ i, active c BM BD i = Bool.true → a ≠ address c BM BD i) :
    (writeOutput s c Out BM BD values).mem r a = s.mem r a := by
  unfold writeOutput
  generalize TileShape.allIndices [BM,BD] = indices
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons]
    split_ifs with hi
    · rw [ih]
      have hn : ¬ (r = Out ∧ a = address c BM BD i) := by
        rintro ⟨hr,ha⟩
        rcases h with hr' | hi'
        · exact hr' hr
        · exact hi' i hi ha
      simp [BlockState.writeMem, hn]
    · exact ih s

theorem readback_list {α : Type} (indices : List α) (s : BlockState) (Out : RegionName)
    (offset : α → Nat) (value : α → ℝ) (mask : α → Bool) (i : α)
    (hnd : indices.Nodup) (hi : i ∈ indices) (hactive : mask i = Bool.true)
    (hinj : ∀ a b, mask a = Bool.true → mask b = Bool.true → offset a = offset b → a = b) :
    (indices.foldl (fun st j => if mask j then st.writeMem Out (offset j) (value j) else st) s).readMem Out (offset i) = value i := by
  obtain ⟨before,after,hlist⟩ := List.append_of_mem hi
  rw [hlist, List.nodup_append, List.nodup_cons] at hnd
  have hnot : i ∉ after := hnd.2.1.1
  rw [hlist, List.foldl_append, List.foldl_cons, hactive, if_pos rfl]
  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem Out offset value
    (fun j => mask j = Bool.true) after _ (offset i) (by
    intro j hj hm he
    have hji := hinj j i hm hactive he.symm
    exact hnot (hji ▸ hj))]
  simp [BlockState.writeMem_readMem]

theorem readback (s : BlockState) (c : Config) (Out : RegionName) (BM BD : Nat)
    (values : TileIndex [BM,BD] → ℝ) (i : TileIndex [BM,BD])
    (hi : active c BM BD i = Bool.true)
    (hinj : ∀ a b, active c BM BD a = Bool.true → active c BM BD b = Bool.true →
      address c BM BD a = address c BM BD b → a = b) :
    (writeOutput s c Out BM BD values).readMem Out (address c BM BD i) = values i :=
  readback_list _ s Out _ _ _ i (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ i) hi hinj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealEpilogue