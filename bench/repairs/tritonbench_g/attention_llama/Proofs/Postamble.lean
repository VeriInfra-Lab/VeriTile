import bench.tritonbench_g.attention_llama.Proofs.RealKernel
import VeriTile.Triton.KernelLemmas.ScatterStore

/-! The existing postamble is specialized only to weaken its full-tile
injectivity condition to actual active stores. This admits ordinary empty_like
layouts whose padded, inactive query lanes overlap active channel offsets. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.Postamble
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
set_option maxHeartbeats 4000000
set_option maxRecDepth 8000

def ActiveInjective (row0 H BM BD som son : Nat) : Prop :=
  ∀ x y : TileIndex [BM,BD], row0+x.1.val < H → row0+y.1.val < H →
    (row0+x.1.val)*som+x.2.1.val*son = (row0+y.1.val)*som+y.2.1.val*son → x=y

theorem store_props (sin : BlockState) (Out : RegionName)
    (oBase som son row0 H BM BD : Nat) (g : TileIndex [BM,BD] → ℝ)
    (hinj : ActiveInjective row0 H BM BD som son) :
    (alOutStoreState sin Out oBase som son row0 H BM BD g).pids = sin.pids ∧
    (∀ a : TileIndex [BM,BD], row0+a.1.val < H →
      (alOutStoreState sin Out oBase som son row0 H BM BD g).readMem Out
        (oBase+(row0+a.1.val)*som+a.2.1.val*son) = g a) := by
  classical
  constructor
  · exact BlockState.foldl_writeMem_prop_masked_pids ..
  · intro a ha
    have h := foldl_store_at (region:=Out)
      (fun i : TileIndex [BM,BD] => oBase+(row0+i.1.val)*som+i.2.1.val*son)
      g (fun i => decide (row0+i.1.val < H)) _ (TileShape.allIndices [BM,BD]) sin a
      (TileShape.mem_allIndices _ a) (by simpa using ha) rfl
      (by
        intro b _ hb he
        apply hinj b a (of_decide_eq_true hb) ha
        exact Nat.add_left_cancel (by simpa only [Nat.add_assoc] using he))
      (TileShape.allIndices_nodup _)
    simpa only [alOutStoreState, decide_eq_true_eq] using h

theorem run (Out : RegionName) (s0 : BlockState)
    (soz soh som son N_HEAD H BM BD : Nat) (sL : BlockState)
    (g : TileIndex [BM, BD] → ℝ)
    (houtinj : ActiveInjective (s0.pids 0 * BM) H BM BD som son)
    (hpids : sL.pids = s0.pids)
    (hacc : sL.regs .real [BM, BD] "acc"
      = some ⟨fun idx : TileIndex [BM, BD] => ((g idx : ℝ) : WithBot ℝ)⟩)
    (hbi : sL.regs .nat [] "batch_id" = some (Tile.scalar (s0.pids 1 / N_HEAD)))
    (hoh : sL.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1 % N_HEAD))) :
    ∃ sP, stepStmts (alPostLoopG Out soz soh som son H BM BD) sL = some sP
      ∧ (∀ idx : TileIndex [BM, BD], alActive s0 H BM idx →
          sP.readMem Out (alOutOffset s0 N_HEAD soz soh som son BM idx) = g idx)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BM, BD], alActive s0 H BM idx →
          o ≠ alOutOffset s0 N_HEAD soz soh som son BM idx) → sP.mem r o = sL.mem r o) := by
  unfold alPostLoopG
  -- W1: start_m = pid0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.programId 0) sL = some (Tile.scalar (s0.pids 0)) from by
      rw [evalOp_programId, hpids]))]
  -- W2: offs_m = start_m * BM + arange BM
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM)) (Op.arange BM)) _
        = some (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)) from by
      rw [evalOp_add, evalOp_arange]
      simp only [evalOp_mul, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext r
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- W3: offs_d = arange BD
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BD _))]
  -- W4: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "batch_id") (Op.constNat soz))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat soh)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat son))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son⟩
              : Tile .nat [BM, BD]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul, evalOp_mul]
      rw [show @evalOp TileDType.nat [BM, 1]
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) _
          = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun r : Fin BM => s0.pids 0 * BM + r.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show @evalOp TileDType.nat [1, BD]
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) _
          = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun d : Fin BD => d.val)))
        from evalOp_expandDim_ref_of_regs _ _ _ _ _ _ (by
          simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
            String.reduceEq, not_false_eq_true])]
      rw [show evalOp (Op.ref .nat [] "batch_id") _ = some (Tile.scalar (s0.pids 1 / N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hbi]
      rw [show evalOp (Op.ref .nat [] "off_hz") _ = some (Tile.scalar (s0.pids 1 % N_HEAD)) from by
        rw [evalOp_ref]
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        exact hoh]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Tile.expandDim_data,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul]
      rfl))]
  -- W5: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BM, BD] "off_o")) _
        = some (alPtrTile Out (alBase s0 N_HEAD soz soh) som son (s0.pids 0 * BM) BM BD) from by
      rw [evalOp_ptrAdd, evalOp_ptrBase]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext fun idx => ?_)
      show (Out, 0 + (alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
        = (Out, alBase s0 N_HEAD soz soh + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son)
      rw [Nat.zero_add]))]
  -- W6: the masked store
  rw [stepStmts.cons_some (alOutStore_step_eq _ Out (alBase s0 N_HEAD soz soh) som son
    (s0.pids 0 * BM) H BM BD g
    (⟨fun idx : TileIndex [BM, BD] => ((g idx : ℝ) : WithBot ℝ)⟩ : Tile .real [BM, BD])
    (fun a => rfl)
    (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
        rw [hacc])
    (by simp only [BlockState.setReg_same])
    (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  obtain ⟨_hpidsP, hread⟩ := store_props _ Out (alBase s0 N_HEAD soz soh) som son
    (s0.pids 0 * BM) H BM BD g houtinj
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hActive
    exact hread idx hActive
  · intro r o houtside
    refine (alOutStore_frame _ Out (alBase s0 N_HEAD soz soh) som son
      (s0.pids 0 * BM) H BM BD g r o houtside).trans ?_
    rfl


end VeriTile.Bench.Repairs.AttentionLlama.Postamble
