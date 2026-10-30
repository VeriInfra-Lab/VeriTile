import bench.tritonbench_g.chunk_gla_fwd.Proofs.SourceBridge

/-! The original consumer's arithmetic proof is valid with masked address
injectivity. This extension reuses its proved primitive steps and K-loop,
strengthening only the final store readback to admit all V tails. The private
primitive theorem names are pinned to the imported source module. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.OutputTail
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open RealOutput


local macro "cgf_vLoad_eq" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_vLoad_eq"))
local macro "cgf_aLoad_eq" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_aLoad_eq"))
local macro "cgf_setReg_mem" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_setReg_mem"))
local macro "cgf_aWhere_run" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_aWhere_run"))
local macro "cgf_finalAcc_run" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_finalAcc_run"))
local macro "cgf_store_eq" : term => pure (Lean.mkIdent (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.num `_private.bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput 0) "VeriTile") "Bench") "TritonBenchG") "ChunkGlaFwd") "RealOutput") "cgf_store_eq"))

noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem post_executes (s0 : BlockState) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t : BlockState)
    (hInj : ∀ i j : TileIndex [BT, BV],
      (tIndex s0 BT i.1.val < T ∧ vIndex s0 BV i.2.1.val < V) →
      (tIndex s0 BT j.1.val < T ∧ vIndex s0 BV j.2.1.val < V) →
      outOffset s0 s_v_h s_v_t BT BV i = outOffset s0 s_v_h s_v_t BT BV j → i = j)
    (hinv : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV
      (numKB K BK) t) :
    ∃ sF, stepStmts (cgfPostLoop v o A s_v_h s_v_t T V BT BV) t = some sF
      ∧ (∀ idx : TileIndex [BT, BV],
          (tIndex s0 BT idx.1.val < T ∧ vIndex s0 BV idx.2.1.val < V) →
          sF.readMem o (outOffset s0 s_v_h s_v_t BT BV idx)
            = cgfOutput s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV idx.1.val idx.2.1.val)
      ∧ BlockState.WriteWithin (footprint o s_v_h s_v_t T V BT BV s0) s0 sF := by
  obtain ⟨-, hmem, hpids, hiv, hit, hibh, hms, hbo, hu⟩ := hinv
  unfold cgfPostLoop
  -- 1-3. the three block pointers
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval v t _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval t "i_bh" (s0.pids 2) s_v_h hibh)
      (cgf_mulConst_eval t "i_t" (s0.pids 1) BT hit)
      (cgf_mulConst_eval t "i_v" (s0.pids 0) BV hiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval _ "i_bh" (s0.pids 2) s_v_h (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (cgf_mulConst_eval _ "i_v" (s0.pids 0) BV (by simpa using hiv))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval A _ _ _ _ [T, BT] [BT, BT] [BT, 1]
      (s0.pids 2 * T * BT) (s0.pids 1 * BT) 0
      (cgf_aBase_eval _ T BT (s0.pids 2) (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (evalOp_constNat 0 _)))]
  -- 4. `b_v = tl.load(p_v, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_vLoad_eq s0 v _ s_v_h s_v_t T V BT BV
      (by simpa [cgf_setReg_mem] using hmem) (by simp) (by simpa using hu)))]
  -- 5. `b_A = tl.load(p_A, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_aLoad_eq s0 A _ T BT (by simpa [cgf_setReg_mem] using hmem) (by simp) (by simpa using hu)))]
  -- 6. `b_A = tl.where(m_s, b_A, 0.)`
  rw [stepStmts.cons_some (cgf_aWhere_run s0 A T BT _ (by simpa using hms) (by simp))]
  -- 7. `b_o += tl.dot(b_A, b_v)`
  rw [stepStmts.cons_some (cgf_finalAcc_run s0 q v g h A s_k_h s_k_t s_v_h s_v_t
    s_h_h s_h_t scale T K V BT BK BV _ (by simpa using hbo) (by simp) (by simp))]
  -- 8. the store
  rw [stepStmts.cons_some
    (cgf_store_eq o s_v_h s_v_t T V BT BV s0 _
      (cgfOutTile s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV)
      (fun idx => cgfOutput s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV idx.1.val idx.2.1.val)
      (fun idx => rfl) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hactive
    unfold cgfStoreState
    exact BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
      (by simpa [tIndex, vIndex] using hactive)
      (fun kk hk heq => hInj kk idx hk hactive heq)
  · intro region off hn
    rw [← store_frame _ _ _ _ _ _ _ _ _ _ region off hn]
    simpa only [cgf_setReg_mem] using (congrFun (congrFun hmem region) off).symm

theorem source_executes
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hInj : ∀ i j : TileIndex [BT, BV],
      (tIndex s BT i.1.val < T ∧ vIndex s BV i.2.1.val < V) →
      (tIndex s BT j.1.val < T ∧ vIndex s BV j.2.1.val < V) →
      outOffset s s_v_h s_v_t BT BV i = outOffset s s_v_h s_v_t BT BV j → i = j) :
    ∃ sF, exec (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BT, BV],
          (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) →
          sF.readMem o (outOffset s s_v_h s_v_t BT BV idx)
            = cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV idx.1.val idx.2.1.val)
      ∧ BlockState.WriteWithin (footprint o s_v_h s_v_t T V BT BV s) s sF := by
  rw [exec, cgf_body_eq, List.append_assoc]
  -- prologue
  obtain ⟨t0, hpre, h0mem, h0pids, h0iv, h0it, h0ibh, h0ms, h0bo, h0u⟩ :=
    cgfPreLoop_run s BT BV
  rw [stepStmts.append_some hpre]
  rw [show [Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
            (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)]
      ++ cgfPostLoop v o A s_v_h s_v_t T V BT BV
    = Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
            (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)
      :: cgfPostLoop v o A s_v_h s_v_t T V BT BV from rfl]
  -- the collapsed K loop
  obtain ⟨t1, hloop, hinv1⟩ :=
    cgfLoop_collapse s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV t0
      ⟨Nat.zero_le _, h0mem, h0pids, h0iv, h0it, h0ibh, h0ms,
        by rw [h0bo, cgfAccTile_zero], h0u.trans hu⟩
  rw [stepStmts.cons_some hloop]
  -- the tail
  obtain ⟨sF, hpost, hout, hframe⟩ :=
    post_executes s q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV t1 hInj hinv1
  exact ⟨sF, hpost, hout, hframe⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.OutputTail
