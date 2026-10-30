import bench.tritonbench_g.attention_llama.Sources.Kernels
import bench.tritonbench_g.attention_llama.Proofs.Mathematical_RealKernel

/-! Local extension of the existing complete source execution proof. The
only changed premise is the V load: masked lanes are source-selected zero.
The existing arithmetic, pointer advance, and memory-frame lemmas are reused. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.SourceExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open VeriTile.Examples.AttentionForwardClosedForm
set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def loopPrefix (sm_scale : ℝ) (N_CTX BM BN BD : Nat) : List Stmt :=
  [ Stmt.assign .nat [BN] "block_n_offs"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n")),
    Stmt.assign .real [BN, BD] "k"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "k_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX)))
          (Op.broadcast (Op.const (0.0 : ℝ)) [BN, BD]))),
    Stmt.assign .real [BM, BN] "qk" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .real [BM, BN] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q")
          (Op.transpose (batch := []) (Op.ref .real [BN, BD] "k")))),
    Stmt.assign .real [BM, BN] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sm_scale)),
    Stmt.assign .real [BM, BN] "qk"
      (Op.where
        (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX)))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) ]

theorem global_score_mask_eval (s : BlockState) (BM BN N_CTX SN : Nat)
    (qktile : Tile .real [BM, BN])
    (hon : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.where
        (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX)))
        (Op.ref .real [BM, BN] "qk") (Op.broadcast Op.negInf [BM, BN])) s
      = some (⟨fun idx => if SN + idx.2.1.val < N_CTX then qktile.data idx else ⊥⟩ : Tile .real [BM, BN]) := by
  have hexp : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => SN + j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have hmask : evalOp (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX))) s
      = some (⟨fun idx : TileIndex [BM, BN] => decide (SN + idx.2.1.val < N_CTX)⟩ : Tile .bool [BM, BN]) := by
    rw [al_evalOp_remap, evalOp_lt]
    simp only [hexp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [hmask, evalOp_ref, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  by_cases hlt : SN + idx.2.1.val < N_CTX
  · simp only [Tile.select_data, hlt, decide_true, if_true]
  · simp only [Tile.select_data, hlt, decide_false, Bool.false_eq_true, if_false]


theorem prefix_steps_with_masked_keys (sc : ℝ) (N_CTX BM BN BD SN : Nat) (sin : BlockState)
    (K : RegionName) (kBase skn skk : Nat)
    (qtile : Tile .real [BM, BD]) (kTfn : TileIndex [BN, BD] → ℝ)
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffn : sin.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hq : sin.regs .real [BM, BD] "q" = some qtile)
    (hkp : sin.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hkload : ∀ idx : TileIndex [BN, BD],
        kTfn idx = if SN + idx.1.val < N_CTX then
          sin.readMem K (kBase + (SN + idx.1.val) * skn + idx.2.1.val * skk) else 0)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sPre, stepStmts (loopPrefix sc N_CTX BM BN BD) sin = some sPre
      ∧ sPre.pids = sin.pids ∧ sPre.mem = sin.mem ∧ (∀ rg o, sPre.undef rg o = 0)
      ∧ sPre.regs .real [BM, BN] "qk" = some (⟨fun idx => if SN + idx.2.1.val < N_CTX then
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
            (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
            (Tile.dot [] qtile (Tile.transpose []
              (⟨fun idx => some (kTfn idx)⟩ : Tile .real [BN, BD]))))
          (Tile.scalar (some sc : WithBot ℝ))).data idx else ⊥⟩ : Tile .real [BM, BN])
      ∧ sPre.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val))
      ∧ sPre.regs .real [BM] "m_prev" = sin.regs .real [BM] "m_prev"
      ∧ sPre.regs .real [BM] "l_prev" = sin.regs .real [BM] "l_prev"
      ∧ sPre.regs .real [BM, BD] "acc" = sin.regs .real [BM, BD] "acc"
      ∧ sPre.regs .real [BM, BD] "q" = sin.regs .real [BM, BD] "q"
      ∧ sPre.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
      ∧ sPre.regs .nat [BN] "offs_n" = sin.regs .nat [BN] "offs_n"
      ∧ sPre.regs .nat [] "batch_id" = sin.regs .nat [] "batch_id"
      ∧ sPre.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz"
      ∧ sPre.regs .ptr [BN, BD] "k_ptrs" = sin.regs .ptr [BN, BD] "k_ptrs"
      ∧ sPre.regs .ptr [BN, BD] "v_ptrs" = sin.regs .ptr [BN, BD] "v_ptrs" := by
  set kT : Tile .real [BN, BD] := ⟨fun idx => some (kTfn idx)⟩ with hkT
  set qkdot : Tile .real [BM, BN] :=
    Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
      (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      (Tile.dot [] qtile (Tile.transpose [] kT)) with hqkdot
  unfold loopPrefix
  -- P1: block_n_offs = start_n + offs_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.ref .nat [BN] "offs_n")) sin
        = some (Tile.vec (fun j : Fin BN => SN + j.val)) from by
      rw [evalOp_add]
      simp only [evalOp_ref, hsn, hoffn, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]))]
  -- P2: k = masked load (actual zero-padded tail values)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "k_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX)))
          (Op.broadcast (Op.const (0.0 : ℝ)) [BN, BD]))) _
        = some kT from by
      rw [al_load_ptr_maskOther_real (Op.ref .ptr [BN, BD] "k_ptrs") _ _ _
        (alPtrTile K kBase skn skk SN BN BD)
        (⟨fun idx : TileIndex [BN, BD] => decide (SN + idx.1.val < N_CTX)⟩ : Tile .bool [BN, BD])
        (⟨fun _ : TileIndex [BN, BD] => some (0.0 : ℝ)⟩ : Tile .real [BN, BD])
        (by rw [evalOp_ref]; simp [hkp])
        (al_colmask_eval _ SN N_CTX (by simp))
        (by simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl)]
      refine congrArg some ?_; ext idx
      by_cases hi : SN + idx.1.val < N_CTX
      · simp only [hi, decide_true, if_true, hkT, hkload idx, BlockState.setReg_readMem]
        rfl
      · simp only [hi, decide_false, Bool.false_eq_true, if_false, hkT, hkload idx]
        norm_num))]
  -- P3: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (al_qkzeros_eval _ BM BN))]
  -- P4: qk += dot(q, trans k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_qk_dot_eval _ BM BN BD (⟨fun _ => some (0:ℝ)⟩) qtile kT
      (by simp) (by simp [hq]) (by simp [hkT])))]
  -- P5: scale only the score graph before imposing negative infinity.
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_qk_scale_eval _ BM BN sc qkdot (by simp [hqkdot])))]
  -- P6: mask using the actual global key, including a partial block.
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (global_score_mask_eval _ BM BN N_CTX SN
      (Tile.bop NumericDType.real.mul Broadcast.scalarR qkdot (Tile.scalar (some sc : WithBot ℝ)))
      (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hqkdot, hkT]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]


theorem tail_steps_with_masked_values (skn svk N_CTX BM BN BD SN : Nat) (sin : BlockState)
    (hBN : 0 < BN)
    (qkT : Tile .real [BM, BN]) (mtile ltile : Tile .real [BM])
    (acctile : Tile .real [BM, BD])
    (K V : RegionName) (kBase vBase skk svn : Nat)
    (vTfn : TileIndex [BN, BD] → ℝ)
    (hqk : sin.regs .real [BM, BN] "qk" = some qkT)
    (hbn : sin.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hmp : sin.regs .real [BM] "m_prev" = some mtile)
    (hlp : sin.regs .real [BM] "l_prev" = some ltile)
    (hacc : sin.regs .real [BM, BD] "acc" = some acctile)
    (hkp : sin.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : sin.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hvload : ∀ idx : TileIndex [BN, BD],
        vTfn idx = if SN + idx.1.val < N_CTX then
          sin.readMem V (vBase + (SN + idx.1.val) * svk + idx.2.1.val * svn) else 0)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ rmaxT : Tile .real [BM],
      Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some rmaxT ∧
      ∃ sF, stepStmts (alLoopTailG skn svk N_CTX BM BN BD) sin = some sF
        ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
        ∧ ∃ (mcurrT lcurrT lrcpT : Tile .real [BM]) (pT : Tile .real [BM, BN])
              (acc1T : Tile .real [BM, BD]),
            mcurrT = Tile.select
                (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rmaxT mtile) rmaxT mtile
            ∧ pT = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
                (Tile.uop WithBot.realExp
                  (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
                    qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT)))
                (Tile.expandDim ⟨1, by simp⟩ lrcpT)
            ∧ lcurrT = Tile.bop NumericDType.real.add Broadcast.nil.consSame
                (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length)
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
                      qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT))))
                (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile
                  (Tile.uop WithBot.realExp
                    (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT)))
            ∧ lrcpT = Tile.bop NumericDType.real.div Broadcast.scalarL
                (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT
            ∧ acc1T = Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acctile
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.bop NumericDType.real.mul Broadcast.nil.consSame
                    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame ltile
                      (Tile.uop WithBot.realExp
                        (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT)))
                    lrcpT))
            ∧ sF.regs .real [BM] "m_prev" = some mcurrT
            ∧ sF.regs .real [BM] "l_prev" = some lcurrT
            ∧ sF.regs .real [BM, BD] "acc" = some
                (Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame
                  acc1T (Tile.dot [] pT (⟨fun idx => some (vTfn idx)⟩ : Tile .real [BN, BD])))
            ∧ sF.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk (SN + BN) BN BD)
            ∧ sF.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn (SN + BN) BN BD)
            ∧ sF.regs .real [BM, BD] "q" = sin.regs .real [BM, BD] "q"
            ∧ sF.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
            ∧ sF.regs .nat [BN] "offs_n" = sin.regs .nat [BN] "offs_n"
            ∧ sF.regs .nat [] "batch_id" = sin.regs .nat [] "batch_id"
            ∧ sF.regs .nat [] "off_hz" = sin.regs .nat [] "off_hz" := by
  set vT : Tile .real [BN, BD] := ⟨fun idx => some (vTfn idx)⟩ with hvT
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkT = some t :=
    ⟨_, by
      unfold Tile.reduceMaxDrop
      rw [dif_pos (show 0 < TileShape.axisDim [BM, BN] (⟨1, by simp⟩ : Fin [BM, BN].length) from hBN)]⟩
  set mcurrT : Tile .real [BM] := Tile.select
      (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame rmaxT mtile) rmaxT mtile
    with hmc
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mtile mcurrT) with hal
  set lprev1T : Tile .real [BM] := Tile.bop NumericDType.real.mul Broadcast.nil.consSame
      ltile alphaT with hlp1
  set pexpT : Tile .real [BM, BN] := Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
        qkT (Tile.expandDim ⟨1, by simp⟩ mcurrT)) with hpexp
  set lcurrT : Tile .real [BM] := Tile.bop NumericDType.real.add Broadcast.nil.consSame
      (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pexpT) lprev1T with hlc
  set lrcpT : Tile .real [BM] := Tile.bop NumericDType.real.div Broadcast.scalarL
      (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) lcurrT with hlr
  set pT : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
      pexpT (Tile.expandDim ⟨1, by simp⟩ lrcpT) with hpT
  set acc1T : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame
      acctile (Tile.expandDim ⟨1, by simp⟩
        (Tile.bop NumericDType.real.mul Broadcast.nil.consSame lprev1T lrcpT)) with hacc1
  refine ⟨rmaxT, hrm, ?_⟩
  simp only [alLoopTailG, Bool.false_eq_true, ↓reduceIte]
  -- T1: m_curr = maximum(max(qk,1), m_prev)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (evalOp_maximumFloat_real_of_where (al_mij_eval _ BM BN mtile qkT rmaxT hmp hqk hrm)))]
  -- T2: l_prev *= exp(m_prev - m_curr)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_alpha_eval _ BM ltile mtile mcurrT (by simp [hlp]) (by simp [hmp]) (by simp [hmc])))]
  -- T3: p = exp(qk - m_curr[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_p_eval _ BM BN Nat.one_lt_two qkT mcurrT (by simp [hqk]) (by simp [hmc])))]
  -- T4: l_curr = sum(p,1) + l_prev
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_lij_eval _ BM BN pexpT lprev1T (by simp [hpexp]) (by simp [hlp1, hal])))]
  -- T5: l_rcp = 1/l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_lrcp_eval _ BM lcurrT (by simp [hlc])))]
  -- T6: p *= l_rcp[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_p_rcp_eval _ BM BN Nat.one_lt_two pexpT lrcpT (by simp [hpexp]) (by simp [hlr])))]
  -- T7: acc *= (l_prev * l_rcp)[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_acc_rescale_eval _ BM BD Nat.one_lt_two acctile lprev1T lrcpT
      (by simp [hacc]) (by simp [hlp1, hal]) (by simp [hlr])))]
  -- T8: p = p (the erased element_ty cast)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM, BN] "p") _ = some pT from by
      rw [evalOp_ref]; simp [hpT]))]
  -- T9: the actual masked load, including partial and wholly masked blocks
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN, BD] "v_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "block_n_offs")) (Op.constNat N_CTX)))
          (Op.broadcast (Op.const (0.0 : ℝ)) [BN, BD]))) _
        = some vT from by
      rw [al_load_ptr_maskOther_real (Op.ref .ptr [BN, BD] "v_ptrs") _ _ _
        (alPtrTile V vBase svk svn SN BN BD)
        (⟨fun idx : TileIndex [BN, BD] => decide (SN + idx.1.val < N_CTX)⟩ : Tile .bool [BN, BD])
        (⟨fun _ : TileIndex [BN, BD] => some (0.0 : ℝ)⟩ : Tile .real [BN, BD])
        (by rw [evalOp_ref]; simp [hvp])
        (al_colmask_eval _ SN N_CTX (by simp [hbn]))
        (by simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl)]
      refine congrArg some ?_; ext idx
      by_cases hi : SN + idx.1.val < N_CTX
      · simp only [hi, decide_true, if_true, hvT, hvload idx, BlockState.setReg_readMem]
        rfl
      · simp only [hi, decide_false, Bool.false_eq_true, if_false, hvT, hvload idx]
        norm_num))]
  -- T10: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_acc_pv_eval _ BM BN BD acc1T pT vT (by simp [hacc1]) (by simp [hpT]) (by simp [hvT])))]
  -- T11: l_prev = l_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM] "l_curr") _ = some lcurrT from by
      rw [evalOp_ref]; simp [hlc]))]
  -- T12: m_prev = m_curr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM] "m_curr") _ = some mcurrT from by
      rw [evalOp_ref]; simp [hmc]))]
  -- T13: k_ptrs += BN * skn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_ptr_advance_eval _ K "k_ptrs" kBase skn skk SN BN BN BD (by simp [hkp])))]
  -- T14: v_ptrs += BN * svk
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (al_ptr_advance_eval _ V "v_ptrs" vBase svk svn SN BN BN BD (by simp [hvp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, mcurrT, lcurrT, lrcpT, pT, acc1T, rfl, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp [hmc]
  · simp [hlc]
  · simp [hacc1, hpT, hvT]
  · simp
  · simp
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]

def sourceTail (skn svk N BM BN BD : Nat) (carryFirst : Bool) : List Stmt :=
  Stmt.assign .real [BM] "m_curr"
    (.max2 Broadcast.nil.consSame
      (.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false (.ref .real [BM, BN] "qk"))
      (.ref .real [BM] "m_prev")) ::
    (alLoopTailG skn svk N BM BN BD carryFirst).drop 1

theorem max_head_step {n : Nat} (x y : Op .real [n])
    (name : RegName) (rest : List Stmt) (s : BlockState) :
    stepStmts (.assign .real [n] name (.max2 (.consSame .nil) x y) :: rest) s =
      stepStmts (.assign .real [n] name (.maximumFloat .real (.consSame .nil) x y) :: rest) s := by
  simp only [stepStmts, stepStmt, FloatDType.toTileDType]
  erw [evalOp_maximumFloat_real]


theorem sourceTail_step (skn svk N BM BN BD : Nat) (carryFirst : Bool) (s : BlockState) :
    stepStmts (sourceTail skn svk N BM BN BD carryFirst) s =
      stepStmts (alLoopTailG skn svk N BM BN BD carryFirst) s := by
  exact max_head_step _ _ "m_curr" ((alLoopTailG skn svk N BM BN BD carryFirst).drop 1) s


def sourceLoop (sm_scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (IS_CAUSAL : Bool) (carryFirst : Bool := Bool.false) : List Stmt :=
  loopPrefix sm_scale N BM BN BD ++
    .ifThen (.constBool IS_CAUSAL) [alCausalWhereStmt SP BM BN] ::
      sourceTail skn svk N BM BN BD carryFirst

section
variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (IS_CAUSAL : Bool)

theorem repaired_body :
    (TypedSource.repaired Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel.body =
      alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
        .ifThen (.constBool IS_CAUSAL) [alBnEndCausalStmt SP BN] ::
        .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
          (sourceLoop sm_scale skn svk N SP BM BN BD IS_CAUSAL) :: alPostLoopG Out soz soh som son H BM BD := by
  simp only [TypedSource.repaired, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
  rfl


/-- The typed source has an actual real algorithm projection. -/
theorem repaired_projection_supported :
    (TypedSource.repaired Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgorithm? =
      .ok (TypedSource.repaired Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel := by
  have h : (TypedSource.repaired Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgorithm? =
      .ok (TypedSource.repaired Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, h]
  simp


end

end VeriTile.Bench.Repairs.AttentionLlama.SourceExecution
