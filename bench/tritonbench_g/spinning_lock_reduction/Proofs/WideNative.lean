import bench.tritonbench_g.spinning_lock_reduction.Proofs.WideAccess

/-! Typed source expression adapter. Native here means the fixed-width source
operations, not a compiler/runtime implementation theorem or GPU execution.
The exact decoder and all finite runs refine the natural controller from
ordinary Buffers and the reached-state control invariant. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

theorem native_decode_eq (h : HostArgs) (buf : Buffers h) (p : Nat) (s : State V)
    (hs : ControlBound h s) : nativeDecode h p s = decode h p s := by
  by_cases hg : p ≥ h.numSms
  · simp [nativeDecode, decode, hg]
  · have hp : p < h.numSms := by omega
    have hp63 : p < 2 ^ 63 := by have := buf.smsGridLimit; omega
    have hc := hs p
    cases hphase : (s.pc p).phase with
    | branch =>
      have hk : h.groupK ≠ 0 := by have := buf.groupPositive; omega
      simp only [nativeDecode, decode, if_neg hg, hphase, if_neg hk,
        reducer64_exact h buf p hp, next64_exact h buf p hp]
    | scan next =>
      have hb : next ≤ h.numSms ∧ reducer h p = true := by simpa [NextBound, hphase] using hc
      simp [nativeDecode, decode, hg, hphase, continues64_exact h buf p next hp hb.1 hb.2]
    | poll next =>
      have hb : next < h.numSms ∧ reducer h p = true := by simpa [NextBound, hphase] using hc
      have hn63 : next < 2 ^ 63 := by have := buf.smsGridLimit; omega
      simp [nativeDecode, decode, hg, hphase, signed64_nat next hn63]
    | gather next lane =>
      have hb : next < h.numSms ∧ reducer h p = true := by simpa [NextBound, hphase] using hc
      by_cases hl : lane < lanes h
      · simp [nativeDecode, decode, hg, hphase, hl, partial_offset64_exact h buf next lane hb.1 hl, pAddress]
      · simp only [nativeDecode, decode, if_neg hg, hphase, if_neg hl, next64_exact h buf next hb.1]
    | publish lane =>
      by_cases hl : lane < lanes h
      · simp [nativeDecode, decode, hg, hphase, hl, partial_offset64_exact h buf p lane hp hl, pAddress]
      · simp [nativeDecode, decode, hg, hphase, hl]
    | release => simp [nativeDecode, decode, hg, hphase, signed64_nat p hp63]
    | output lane =>
      by_cases hl : lane < lanes h
      · have hm : outputMask64 h p lane = mask h p lane := by
          apply Bool.eq_iff_iff.mpr
          simpa [mask] using mask64_exact h buf p lane hp hl
        by_cases hmask : outputMask64 h p lane = true
        · have hoff := output_offset64_exact h buf p lane hp hl hmask
          simp [nativeDecode, decode, hg, hphase, hl, ← hm, hmask, hoff, cAddress]
        · simp [nativeDecode, decode, hg, hphase, hl, ← hm, hmask]
      · simp [nativeDecode, decode, hg, hphase, hl]
    | done => simp [nativeDecode, decode, hg, hphase]
    | fault => simp [nativeDecode, decode, hg, hphase]

theorem native_step_eq (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (p : Nat) (s : State V) (hs : ControlBound h s) : nativeStep b h p s = step b h p s := by
  unfold nativeStep step
  rw [native_decode_eq h buf p s hs]

theorem native_run_eq (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (s : State V) (hs : ControlBound h s) : nativeRun b h schedule s = run b h schedule s := by
  induction schedule generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [nativeRun, run, native_step_eq b h buf p s hs]
    rw [ih _ (step_controlBound h b s p hs)]

theorem native_launch_eq (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (mem : Memory V) :
    nativeRun b h schedule (initial b mem) = run b h schedule (initial b mem) :=
  native_run_eq b h buf schedule _ (by intro p; trivial)

#axiomsClean native_decode_eq
#axiomsClean native_launch_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide
