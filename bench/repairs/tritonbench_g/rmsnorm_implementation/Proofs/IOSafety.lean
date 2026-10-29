import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.ActiveOutput
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Source
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Safety

/-! Actual evaluated access safety of both complete passes. Reuses the existing
per-body walks and invariants with active-only output injectivity. This retains
empty/singleton domains, zero singleton strides and arbitrary undefined loads. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.IOSafety
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedVariables false

theorem legacy (R : RoundingModel) (bounds : RegionBounds)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hxo : x ≠ o) (hwo : w ≠ o) (s : BlockState)
    (unique : Function.Injective (fun k : Fin N => outColOffset s sob som sok k.val))
    (hbx : ∀ k, k<N → s.pids 0*sxb+s.pids 1*sxm+k*sxk < bounds x)
    (hbw : ∀ k, k<N → k*srw < bounds w)
    (hbo : ∀ k, k<N → s.pids 0*sob+s.pids 1*som+k*sok < bounds o) :
    (rmsnorm_implementation x w o sxb sxm sxk srw sob som sok N B eps).toAlgKernel.TraceSafeR R bounds s := by
  have hStepNe : B ≠ 0 := Nat.ne_of_gt hB
  unfold Kernel.TraceSafeR
  rw [rmsnorm_implementation_toAlg_body]
  simp only [List.append_assoc, List.singleton_append]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [rmsVarPreLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    rw [rmsImplPre_castFree R sxb sxm B s] at hs1
    have hCtx0 :=
      rmsVarLoopContextInvariant_init_of_preloop s s1 x sxb sxm sxk N B hs1
    obtain ⟨hPids0, hPb0, hPm0, hRw0⟩ :=
      rmsVarPreLoop_step_preserves_pids_read s s1 w sxb sxm B hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- pass 1 (invariant principle over `rmsVarLoopContextInvariant`)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_strart_ptr" N B
        (rmsVarLoopBody x sxk N B)
        (fun off st => rmsVarLoopContextInvariant s x sxb sxm sxk N B off st)
        ?_ 0 s1 hCtx0
      intro i stt hi hP
      have hbN' : stt.regs .nat [B] "block_n_size"
          = some (Tile.vec (fun j : Fin B => j.val)) := hP.2.2.1
      refine ⟨variance_body_safe R bounds x sxb sxm sxk N B s stt i hbN' hP.2.1
        (fun j hj => hbx (i+j.val) hj), ?_⟩
      obtain ⟨st', hstep, hCtx'⟩ :=
        rmsVarLoopContextInvariant_body_step_exists s stt x sxb sxm sxk N B i hP
      exact ⟨st', by rw [rmsImplVarBody_castFree]; exact hstep, hCtx'⟩
    · intro s2 hs2
      -- identify the pass-1 exit state via the exact loop run
      obtain ⟨finalV, sv, hsv, hVarFinal, hPV⟩ :=
        forRange_inv (idx := "block_n_strart_ptr") (start := 0) (stop := N) (step := B)
          (body := rmsVarLoopBody x sxk N B)
          (P := fun off st =>
            rmsVarLoopContextInvariant s x sxb sxm sxk N B off st ∧
            st.pids = s.pids ∧
            st.regs .nat [] "pid_batch" = some (Tile.scalar (s.pids 0)) ∧
            st.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 1)) ∧
            (∀ offset, st.readMem w offset = s.readMem w offset))
          hStepNe ⟨hCtx0, hPids0, hPb0, hPm0, hRw0⟩
          (fun i stt hlt hP => by
            obtain ⟨st', hstep, hCtx'⟩ :=
              rmsVarLoopContextInvariant_body_step_exists s stt x sxb sxm sxk N B i hP.1
            obtain ⟨hPidsS, hPbS, hPmS, hRwS⟩ :=
              rmsVarLoopBody_step_preserves_pids_read s stt st' x w sxb sxm sxk N B i hP.1 hstep
            refine ⟨st', hstep, hCtx', ?_, ?_, ?_, ?_⟩
            · rw [hPidsS]; exact hP.2.1
            · rw [hPbS]; exact hP.2.2.1
            · rw [hPmS]; exact hP.2.2.2.1
            · intro offset; rw [hRwS offset]; exact hP.2.2.2.2 offset)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (rmsImplVarBody_castFree R x sxk N B) "block_n_strart_ptr",
        ← stepForRangeAux.forRange_unfold, hsv] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨hVarCtx, hVarPids, hVarPb, hVarPm, hVarRw⟩ := hPV
      refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · -- reduce/sqrt tail: register-only assigns
        refine Stmt.TraceSafeListR.of_forall _ _ ?_
        intro stmt hst s'
        simp only [rmsStdPostLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
      · intro s3 hs3
        rw [rmsImplPost_castFree R N B eps sv] at hs3
        have hOutBase :=
          rmsStdPostLoop_step_to_out_init s sv s3 x w o sxb sxm sxk srw sob som sok
            N B finalV eps hVarCtx hVarFinal hB hVarPids hVarPb hVarPm hVarRw hs3
        -- pass 2 (invariant principle over `rmsOutLoopContextInvariant` + alignment)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 _ => Stmt.TraceSafeListR.nil_intro)
        simp only [Stmt.TraceSafeR]
        refine Stmt.forRangeTraceSafeR_inv R bounds "block_n_strart_ptr" N B
          (rmsOutLoopBody x w o sxk srw sob som sok N B)
          (fun off st => off % B = 0 ∧
            rmsOutLoopContextInvariant s x w o sxb sxm sxk srw sob som sok N B off eps st)
          ?_ 0 s3 ⟨Nat.zero_mod B, hOutBase⟩
        intro i stt hi hQ
        obtain ⟨hMod, hCtx⟩ := hQ
        have hQd := hCtx
        obtain ⟨hOutInv, hPidsQ, hPbQ, hPmQ, hOmQ, hBnQ, hStdQ, hRXQ, hRWQ⟩ := hQd
        have hbN' : stt.regs .nat [B] "block_n_size"
            = some (Tile.vec (fun j : Fin B => j.val)) := hBnQ
        refine ⟨output_body_safe R bounds x w o sxb sxm sxk srw sob som sok N B s stt i
            hPbQ hPmQ hbN' hOmQ
            (fun j hj => hbx (i+j.val) hj)
            (fun j hj => hbw (i+j.val) hj)
            (fun j hj => hbo (i+j.val) hj), ?_⟩
        obtain ⟨st', hstep, hCtx'⟩ :=
          ActiveOutput.body_exists s stt x w o sxb sxm sxk srw
            sob som sok N B i eps hCtx hxo hwo unique
        exact ⟨st', by rw [rmsImplOutBody_castFree]; exact hstep,
          rmsVarLoopOffset_mod_step i B hMod, hCtx'⟩


theorem original (bounds : RegionBounds)
    (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) (s : BlockState)
    (unique : Function.Injective (fun k : Fin n => outColOffset s sob som sok k.val))
    (hx : ∀ k, k<n → s.pids 0*sxb+s.pids 1*sxm+k*sxk < bounds "x")
    (hw : ∀ k, k<n → k*sw < bounds "weights")
    (ho : ∀ k, k<n → s.pids 0*sob+s.pids 1*som+k*sok < bounds "out") :
    (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  rw [Source.original]
  apply (Safety.typed_safe_iff bounds .fp32 .fp32 .fp32 Bool.false _ _ _ _ _ _ _ _ _ _ _ _ _ _).mpr
  exact legacy .triv bounds _ _ _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide) s unique hx hw ho

theorem commuted (bounds : RegionBounds)
    (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) (s : BlockState)
    (unique : Function.Injective (fun k : Fin n => outColOffset s sob som sok k.val))
    (hx : ∀ k, k<n → s.pids 0*sxb+s.pids 1*sxm+k*sxk < bounds "x")
    (hw : ∀ k, k<n → k*sw < bounds "weights")
    (ho : ∀ k, k<n → s.pids 0*sob+s.pids 1*som+k*sok < bounds "out") :
    (indexCommutedKernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  rw [Source.commuted]
  apply (Safety.typed_safe_iff bounds .fp32 .fp32 .fp32 Bool.true _ _ _ _ _ _ _ _ _ _ _ _ _ _).mpr
  exact legacy .triv bounds _ _ _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide) s unique hx hw ho

#axiomsClean original
#axiomsClean commuted
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.IOSafety
