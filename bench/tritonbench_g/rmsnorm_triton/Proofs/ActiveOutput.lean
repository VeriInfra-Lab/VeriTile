import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_Real
import VeriTile.Meta.StatementAudit

/-! The established two-pass invariant proof, generalized from positive output
stride to injectivity only on active output coordinates. A singleton may have
zero stride. No numerical or loop invariant is changed. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton.ActiveOutput
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Full Full.ScratchRms Full.Wb Full.VarLoop Full.VarFinal Full.Mean
  Full.MathHeart Full.PostStep Full.Prefix Full.WbLoop Full.WbFinal
  Full.WbBody Full.StoreLemma Full.Final Full.Final2 Finset
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem rmsPrefix_castFree (R : RoundingModel) (x : RegionName)
    (sxb sxm B : Nat) (t : BlockState) :
    stepStmtsR R (prefixStmts x sxb sxm B) t
      = stepStmts (prefixStmts x sxb sxm B) t := by
  simp only [prefixStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-1 body is cast-free (masked `.real` load, `.real → .real`
casts, a real multiply-add): it steps identically under `stepStmtsR R`, so
the exact `varInv` stack transports to `execR`. -/
private theorem rmsVarBody_castFree (R : RoundingModel) (x : RegionName)
    (sxk N B : Nat) (t : BlockState) :
    stepStmtsR R (varBody x sxk N B) t = stepStmts (varBody x sxk N B) t := by
  simp only [varBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The reduce/rsqrt tail is cast-free. -/
private theorem rmsPost_castFree (R : RoundingModel) (N B : Nat) (eps : ℝ)
    (t : BlockState) :
    stepStmtsR R (postStmts N B eps) t = stepStmts (postStmts N B eps) t := by
  simp only [postStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The pass-2 body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped` (`writeMemTypedR R .real` is definitionally the exact
write), so the whole storing loop steps identically under `stepStmtsR R`
and the exact `wbInv` stack transports to `execR`. -/
private theorem rmsWbBody_castFree (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (t : BlockState) :
    stepStmtsR R (wbBody x w o sxb sxm sxk srw sob som sok N B) t
      = stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B) t := by
  simp only [wbBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl


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

/-- Cell-level frame of a `Bool`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling
of the shared `foldl_store_preserve`). -/
private theorem foldl_writeMem_bool_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool)
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, mask k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      cases hm : mask hd
      · simp only [hm, Bool.false_eq_true, if_false]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk
      · simp only [hm, if_true]
        rw [ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk,
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self (by rw [hm]))

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one pass-2 iteration** (the `mem` twin of
`wbStep`, same walk): from the `wbInv` register pins, one storing body
iteration leaves every cell off the `{(out, outOff k) : k < N_SIZE}`
write window untouched — the masked scatter store only hits active lanes
`i + j < N_SIZE`, whose offsets are `outOff (i + j)`. -/
private theorem rms_wbBody_step_frame (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat)
    (s0 st st' : BlockState) (RS : ℝ) (i : Nat) (hiB : B ∣ i)
    (hpids : st.pids = s0.pids)
    (hrmx : st.readMem x = s0.readMem x) (hrmw : st.readMem w = s0.readMem w)
    (hpb : st.regs .nat [] "pid_batch" = some (Tile.scalar (s0.pids 0)))
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 1)))
    (hblockN : st.regs .nat [B] "block_N" = some (Tile.vec (fun j : Fin B => j.val)))
    (hoffsm : st.regs .nat [] "offs_m" = some (Tile.scalar (s0.pids 0 * sxb + s0.pids 1 * sxm)))
    (hrstd : st.regs .real [] "rstd" = some (Tile.scalar (some RS)))
    (hstep : stepStmts (wbBody x w o sxb sxm sxk srw sob som sok N B)
        (st.setReg "block_n_start_idx" .nat [] (Tile.scalar i)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s0 sob som sok k.val) :
    st'.mem r oo = st.mem r oo := by
  exact VeriTile.Bench.TritonBenchG.RmsnormTriton.rms_wbBody_step_frame
    x w o sxb sxm sxk srw sob som sok N B s0 st st' RS i hiB hpids hrmx hrmw
    hpb hpm hblockN hoffsm hrstd hstep r oo hcond


theorem run (R : RoundingModel) (x w o : RegionName)
    (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w)
    (s₀ : BlockState)
    (hinj : Function.Injective (fun k : Fin N => outOff s₀ sob som sok k.val)) :
    ∃ sfin,
      execR R (rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀ = some sfin
      ∧ (∀ k : Fin N, sfin.readMem o (outOff s₀ sob som sok k.val)
          = rmsSpecFull s₀ x w sxb sxm sxk srw N eps k.val)
      ∧ (∀ r oo, (r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s₀ sob som sok k.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  -- prologue
  obtain ⟨sp, hsp, hP0⟩ := preLoop x sxb sxm sxk N B s₀
  have hspMem : sp.mem = s₀.mem :=
    stepStmts_assigns_mem (prefixStmts x sxb sxm B)
      (by
        intro stmt hst
        simp only [prefixStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hsp
  -- pass 1 with the memory frame carried alongside `varInv`
  obtain ⟨finalv, sv, hsv, hNlev, hPV⟩ :=
    forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
      (body := varBody x sxk N B)
      (P := fun i stt => varInv x sxb sxm sxk N B s₀ i stt ∧ stt.mem = s₀.mem)
      (Nat.pos_iff_ne_zero.mp hB) ⟨hP0, hspMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hinv'⟩ := varInv_step x sxb sxm sxk N B s₀ hB i stt hlt hP.1
        refine ⟨st', hstep, hinv', ?_⟩
        have hmm : st'.mem = (stt.setReg "block_n_start_idx" .nat [] (Tile.scalar i)).mem :=
          stepStmts_assigns_mem (varBody x sxk N B)
            (by
              intro stmt hst
              simp only [varBody, List.mem_cons, List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        rw [hmm]
        exact hP.2)
  obtain ⟨⟨hpidsV, hrmV, hdvdV, hpbV, hpmV, hbNV, homV, hvarV⟩, hmemV⟩ := hPV
  -- reduce/rsqrt tail
  obtain ⟨sr, hsr, hrstdR, hpresR, hpidsR, hrmR⟩ :=
    postStep s₀ x sxb sxm sxk N B (finalv / B) eps sv hvarV
  have hsrMem : sr.mem = s₀.mem := by
    rw [stepStmts_assigns_mem (postStmts N B eps)
      (by
        intro stmt hst
        simp only [postStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hsr]
    exact hmemV
  -- the `wbInv` base at the post state (mirrors the exact headline)
  have hcvge : N ≤ finalv / B * B := by
    have hdvd : finalv / B * B = finalv := Nat.div_mul_cancel hdvdV
    omega
  have hmean : (∑ j : Fin B, varAcc s₀ x sxb sxm sxk N B (finalv / B) j) / (N:ℝ)
      = meanSq s₀ x sxb sxm sxk N := by
    unfold meanSq
    rw [varAcc_sum_eq s₀ x sxb sxm sxk N B (finalv / B) hB hcvge]
  set RS : ℝ := WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps))) with hRSdef
  have hRS0 : sr.regs .real [] "rstd" = some (Tile.scalar (WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps)))) := by
    rw [hrstdR, hmean]
  have hRSsome : WithBot.realRsqrt (some (meanSq s₀ x sxb sxm sxk N + eps)) = some RS := by
    rw [hRSdef]; simp [WithBot.realRsqrt]
  have hwbBase : wbInv x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ 0 sr := by
    refine ⟨?_, ⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hpidsR, hpidsV]
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpbV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hpmV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact hbNV
    · rw [hpresR _ _ _ (by decide) (by decide)]; exact homV
    · rw [hRS0, hRSsome]
    · funext ofs; rw [congrFun (congrFun hrmR x) ofs, congrFun (congrFun hrmV x) ofs]
    · funext ofs; rw [congrFun (congrFun hrmR w) ofs, congrFun (congrFun hrmV w) ofs]
    · intro k; rw [if_neg (by omega)]
      rw [congrFun (congrFun hrmR o) _, congrFun (congrFun hrmV o) _]
  -- pass 2 with the conditional cell frame carried alongside `wbInv`
  obtain ⟨finalw, sw, hsw, hNlew, hQW⟩ :=
    forRange_inv (idx := "block_n_start_idx") (start := 0) (stop := N) (step := B)
      (body := wbBody x w o sxb sxm sxk srw sob som sok N B)
      (P := fun i stt => wbInv x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ i stt
        ∧ ∀ r oo, (r ≠ o ∨ ∀ k : Fin N, oo ≠ outOff s₀ sob som sok k.val) →
            stt.mem r oo = s₀.mem r oo)
      (Nat.pos_iff_ne_zero.mp hB)
      ⟨hwbBase, fun r oo _ => by rw [hsrMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hinv'⟩ := wbInv_step x w o sxb sxm sxk srw sob som sok N B s₀ RS s₀ hB hinj hox how i stt hlt hQ.1
        refine ⟨st', hstep, hinv', ?_⟩
        intro r oo hcond
        obtain ⟨hpidsQ, hiBQ, hpbQ, hpmQ, hbNQ, homQ, hrsQ, hrxQ, hrwQ, hmemQ⟩ := hQ.1
        rw [rms_wbBody_step_frame x w o sxb sxm sxk srw sob som sok N B s₀ stt st' RS i
          hiBQ hpidsQ hrxQ hrwQ hpbQ hpmQ hbNQ homQ hrsQ hstep r oo hcond]
        exact hQ.2 r oo hcond)
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, hreadW⟩, hframeW⟩ := hQW
  -- assemble the `execR` run through the cast-free collapses
  have hVarR : stepStmtR R (Stmt.forRange "block_n_start_idx" 0 N B (varBody x sxk N B)) sp = some sv := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsVarBody_castFree R x sxk N B) "block_n_start_idx",
      ← stepForRangeAux.forRange_unfold]
    exact hsv
  have hWbR : stepStmtR R (Stmt.forRange "block_n_start_idx" 0 N B
      (wbBody x w o sxb sxm sxk srw sob som sok N B)) sr = some sw := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (rmsWbBody_castFree R x w o sxb sxm sxk srw sob som sok N B) "block_n_start_idx",
      ← stepForRangeAux.forRange_unfold]
    exact hsw
  refine ⟨sw, ?_, ?_, ?_⟩
  · show execR R (rmsnorm_triton x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel s₀ = some sw
    unfold execR
    rw [body_decomp x w o sxb sxm sxk srw sob som sok N B eps]
    simp only [List.append_assoc, List.singleton_append, List.cons_append,
      List.nil_append]
    rw [stepStmtsR_append R (prefixStmts x sxb sxm B) _ s₀,
      rmsPrefix_castFree R x sxb sxm B s₀, hsp, Option.bind_some,
      stepStmtsR_cons_some hVarR,
      stepStmtsR_append R (postStmts N B eps)
        [Stmt.forRange "block_n_start_idx" 0 N B (wbBody x w o sxb sxm sxk srw sob som sok N B)] sv,
      rmsPost_castFree R N B eps sv, hsr,
      Option.bind_some, stepStmtsR_cons_some hWbR, stepStmtsR_nil]
  · intro k
    rw [hreadW k, if_pos (lt_of_lt_of_le k.isLt hNlew)]
    rfl
  · exact hframeW


#axiomsClean run
end
end VeriTile.Bench.Repairs.RmsnormTriton.ActiveOutput
