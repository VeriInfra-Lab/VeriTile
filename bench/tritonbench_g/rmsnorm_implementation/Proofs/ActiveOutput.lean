import bench.tritonbench_g.rmsnorm_implementation.Proofs.Mathematical_Real
import VeriTile.Meta.StatementAudit

/-! Active-output injectivity is sufficient. In particular a single logical
column may have zero output stride; masked tail addresses need not be unique.
The two-pass invariants and arithmetic proofs are reused unchanged. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.ActiveOutput
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem write_current
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ) (i : Fin BLOCK_N_SIZE)
    (hActive : off + i.val < N_SIZE)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hOutInj : Function.Injective
      (fun col : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k col.val))
    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    st'.readMem out_ptr
        (outColOffset s0 stride_out_batch stride_out_m stride_out_k
          (off + i.val)) =
      rmsnormWeightedYFullNSpec s0 x_ptr rms_w_ptr stride_x_batch stride_x_m
        stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps ⟨off + i.val, hActive⟩ := by
  unfold rmsOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
    hReadX, hReadW, outColOffset, rmsStdFullNSpec] at hStep
  subst st'
  simp only [outColOffset]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _
    (fun idx : TileIndex [BLOCK_N_SIZE] => s0.pids 0 * stride_out_batch +
      s0.pids 1 * stride_out_m + (off + idx.1.val) * stride_out_k) _
    (fun idx : TileIndex [BLOCK_N_SIZE] => off + idx.1.val < N_SIZE)
    ((i, PUnit.unit) : TileIndex [BLOCK_N_SIZE]) hActive]
  · simp [hActive, rmsnormWeightedYFullNSpec, rmsnormYFullNSpec, rmsInvVarFullN,
      rmsStdFullNSpec, outColOffset, div_eq_mul_inv]
  · intro idx active same
    have equal := hOutInj (a₁ := ⟨off+idx.1.val, active⟩) (a₂ := ⟨off+i.val, hActive⟩) same
    have lanes : idx.1 = i := Fin.ext (by have := congrArg Fin.val equal; simp only at this; omega)
    cases idx with
    | mk lane unit => cases unit; simp only at lanes; subst lane; rfl


theorem output_invariant
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hInv : rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps off st)
    (hPidBatch :
      st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hPidM :
      st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hOffsetM :
      st.regs .nat [] "offset_m" =
        some (Tile.scalar
          (s0.pids 0 * stride_x_batch + s0.pids 1 * stride_x_m)))
    (hBlockN :
      st.regs .nat [BLOCK_N_SIZE] "block_n_size" =
        some { data := fun idx : TileIndex [BLOCK_N_SIZE] => idx.1.val })
    (hStd :
      st.regs .real [] "std" =
        some (Tile.scalar
          (rmsStdFullNSpec s0 x_ptr stride_x_batch stride_x_m stride_x_k
            N_SIZE BLOCK_N_SIZE eps)))
    (hReadX : ∀ offset, st.readMem x_ptr offset = s0.readMem x_ptr offset)
    (hReadW : ∀ offset, st.readMem rms_w_ptr offset = s0.readMem rms_w_ptr offset)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))

    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    rmsnormOutLoopInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps
      (off + BLOCK_N_SIZE) st' := by
  intro col hWritten
  by_cases hOld : col.val < off
  · rw [rmsOutLoopBody_step_preserves_old_output s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      col hOld hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hOutInj hStep]
    exact hInv col hOld
  · have hOffLe : off ≤ col.val := Nat.le_of_not_gt hOld
    let lane : Fin BLOCK_N_SIZE := ⟨col.val - off, by omega⟩
    have hLaneActive : off + lane.val < N_SIZE := by
      have hcol : off + lane.val = col.val := by
        simp [lane]
        omega
      simp [hcol, col.isLt]
    have hcolEq : off + lane.val = col.val := by
      simp [lane]
      omega
    have hRead := write_current s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      lane hLaneActive hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW
      hOutInj hStep
    simpa [hcolEq] using hRead

theorem context_step
    (s0 st st' : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps st)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))

    (hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st') :
    rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
      (off + BLOCK_N_SIZE) eps st' := by
  rcases hCtx with
    ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
  have hOutInv' :=
    output_invariant s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hOutInv hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hOutInj
      hStep
  rcases rmsOutLoopBody_step_preserves_regs_pids s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hPids hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hStep with
    ⟨hPids', hPidBatch', hPidM', hOffsetM', hBlockN', hStd'⟩
  rcases rmsOutLoopBody_step_preserves_reads s0 st st' x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
      hPidBatch hPidM hOffsetM hBlockN hStd hReadX hReadW hXOutNe hWOutNe hStep with
    ⟨hReadX', hReadW'⟩
  exact ⟨hOutInv', hPids', hPidBatch', hPidM', hOffsetM', hBlockN', hStd', hReadX', hReadW'⟩

theorem body_exists
    (s0 st : BlockState) (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
      stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps st)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hOutInj : Function.Injective
      (fun i : Fin N_SIZE =>
        outColOffset s0 stride_out_batch stride_out_m stride_out_k i.val))
 :
    ∃ st',
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) = some st' ∧
      rmsOutLoopContextInvariant s0 x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        (off + BLOCK_N_SIZE) eps st' := by
  rcases hCtx with
    ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
  cases hStep :
      stepStmts
        (rmsOutLoopBody x_ptr rms_w_ptr out_ptr stride_x_k stride_rms_w
          stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE)
        (st.setReg "block_n_strart_ptr" .nat [] (Tile.scalar off)) with
  | none =>
      unfold rmsOutLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, Tile.bop,
        Tile.cop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind, FloatDType.cast,
        hReadX, hReadW, hPids, outColOffset, rmsStdFullNSpec] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact context_step s0 st st' x_ptr rms_w_ptr
        out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE off eps
        ⟨hOutInv, hPids, hPidBatch, hPidM, hOffsetM, hBlockN, hStd, hReadX, hReadW⟩
        hXOutNe hWOutNe hOutInj hStep

private theorem stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

theorem run (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hxo : x ≠ o) (hwo : w ≠ o) (s₀ : BlockState)
    (hOutInj : Function.Injective (fun i : Fin N => outColOffset s₀ sob som sok i.val)) :
    ∃ sfin,
      execR R (rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀
        = some sfin
      ∧ (∀ i : Fin N, sfin.readMem o (outColOffset s₀ sob som sok i.val)
          = rmsnormWeightedYFullNSpec s₀ x w sxb sxm sxk srw N B eps i)
      ∧ (∀ r oo, (r ≠ o ∨ ∀ i : Fin N, oo ≠ outColOffset s₀ sob som sok i.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hStepNe : B ≠ 0 := Nat.ne_of_gt hB
  -- prologue
  obtain ⟨stPre, hPre⟩ := rmsImplPre_exists sxb sxm B s₀
  have hPreCtx :=
    rmsVarLoopContextInvariant_init_of_preloop s₀ stPre x sxb sxm sxk N B hPre
  obtain ⟨hPrePids, hPrePb, hPrePm, hPreRw⟩ :=
    rmsVarPreLoop_step_preserves_pids_read s₀ stPre w sxb sxm B hPre
  have hPreMem : stPre.mem = s₀.mem :=
    stepStmts_assigns_mem (rmsVarPreLoop sxb sxm B)
      (by
        intro stmt hst
        simp only [rmsVarPreLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hPre
  -- pass 1 with the memory frame carried alongside the context invariant
  obtain ⟨finalV, stVar, hVarFor, hVarFinal, hVarP⟩ :=
    forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
      (body := rmsVarLoopBody x sxk N B)
      (P := fun off st =>
        rmsVarLoopContextInvariant s₀ x sxb sxm sxk N B off st ∧
        st.pids = s₀.pids ∧
        st.regs .nat [] "pid_batch" = some (Tile.scalar (s₀.pids 0)) ∧
        st.regs .nat [] "pid_m" = some (Tile.scalar (s₀.pids 1)) ∧
        (∀ offset, st.readMem w offset = s₀.readMem w offset) ∧
        st.mem = s₀.mem)
      hStepNe
      ⟨hPreCtx, hPrePids, hPrePb, hPrePm, hPreRw, hPreMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hCtx'⟩ :=
          rmsVarLoopContextInvariant_body_step_exists s₀ stt x sxb sxm sxk N B i hP.1
        obtain ⟨hPidsS, hPbS, hPmS, hRwS⟩ :=
          rmsVarLoopBody_step_preserves_pids_read s₀ stt st' x w sxb sxm sxk N B i hP.1 hstep
        have hMemS : st'.mem
            = (stt.setReg "block_n_strart_ptr" .nat [] (Tile.scalar i)).mem :=
          stepStmts_assigns_mem (rmsVarLoopBody x sxk N B)
            (by
              intro stmt hst
              simp only [rmsVarLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        refine ⟨st', hstep, hCtx', ?_, ?_, ?_, ?_, ?_⟩
        · rw [hPidsS]; exact hP.2.1
        · rw [hPbS]; exact hP.2.2.1
        · rw [hPmS]; exact hP.2.2.2.1
        · intro offset; rw [hRwS offset]; exact hP.2.2.2.2.1 offset
        · rw [hMemS]; exact hP.2.2.2.2.2)
  obtain ⟨hVarCtx, hVarPids, hVarPb, hVarPm, hVarRw, hVarMem⟩ := hVarP
  -- reduce/sqrt tail
  obtain ⟨stStd, hStd⟩ := rmsImplPost_exists N B eps stVar _ hVarCtx.1.2
  have hOutBase :=
    rmsStdPostLoop_step_to_out_init s₀ stVar stStd x w o sxb sxm sxk srw sob som sok
      N B finalV eps hVarCtx hVarFinal hB hVarPids hVarPb hVarPm hVarRw hStd
  have hStdMem : stStd.mem = s₀.mem := by
    rw [stepStmts_assigns_mem (rmsStdPostLoop N B eps)
      (by
        intro stmt hst
        simp only [rmsStdPostLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hStd]
    exact hVarMem
  -- pass 2 with the conditional cell frame carried alongside the invariant
  obtain ⟨finalO, stOut, hOutFor, hOutFinal, hOutP⟩ :=
    forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
      (body := rmsOutLoopBody x w o sxk srw sob som sok N B)
      (P := fun off st =>
        rmsOutLoopContextInvariant s₀ x w o sxb sxm sxk srw sob som sok N B off eps st ∧
        ∀ r oo, (r ≠ o ∨ ∀ i : Fin N, oo ≠ outColOffset s₀ sob som sok i.val) →
          st.mem r oo = s₀.mem r oo)
      hStepNe
      ⟨hOutBase, fun r oo _ => by rw [hStdMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hQ'⟩ :=
          body_exists s₀ stt x w o sxb sxm sxk srw
            sob som sok N B i eps hQ.1 hxo hwo hOutInj
        refine ⟨st', hstep, hQ', ?_⟩
        intro r oo hcond
        obtain ⟨hOutInv, hPidsQ, hPbQ, hPmQ, hOmQ, hBnQ, hStdQ, hRXQ, hRWQ⟩ := hQ.1
        rw [rmsImplOutBody_step_frame s₀ stt st' x w o sxb sxm sxk srw sob som sok N B i eps
          hPbQ hPmQ hOmQ hBnQ hStdQ hRXQ hRWQ hstep r oo hcond]
        exact hQ.2 r oo hcond)
  -- assemble the `execR` run through the cast-free collapses
  have hVarR : stepStmtR R (Stmt.forRange "block_n_strart_ptr" 0 N B
      (rmsVarLoopBody x sxk N B)) stPre = some stVar := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsImplVarBody_castFree R x sxk N B) "block_n_strart_ptr",
      ← stepForRangeAux.forRange_unfold]
    exact hVarFor
  have hOutR : stepStmtR R (Stmt.forRange "block_n_strart_ptr" 0 N B
      (rmsOutLoopBody x w o sxk srw sob som sok N B)) stStd = some stOut := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsImplOutBody_castFree R x w o sxk srw sob som sok N B) "block_n_strart_ptr",
      ← stepForRangeAux.forRange_unfold]
    exact hOutFor
  refine ⟨stOut, ?_, ?_, ?_⟩
  · show execR R (rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀
        = some stOut
    unfold execR
    rw [rmsnorm_implementation_toAlg_body]
    simp only [List.append_assoc, List.singleton_append, List.cons_append,
      List.nil_append]
    rw [stepStmtsR_append R (rmsVarPreLoop sxb sxm B) _ s₀,
      rmsImplPre_castFree R sxb sxm B s₀, hPre, Option.bind_some,
      stepStmtsR_cons_some hVarR,
      stepStmtsR_append R (rmsStdPostLoop N B eps)
        [Stmt.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody x w o sxk srw sob som sok N B)] stVar,
      rmsImplPost_castFree R N B eps stVar, hStd, Option.bind_some,
      stepStmtsR_cons_some hOutR, stepStmtsR_nil]
  · intro i
    exact hOutP.1.1 i (Nat.lt_of_lt_of_le i.isLt hOutFinal)
  · exact hOutP.2
#axiomsClean run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.ActiveOutput
