import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Candidate
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressSchedule
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem setReg_comm (s : State α) (a b : RegName)
    (da db : TileDType) (sha shb : TileShape)
    (va : Values α da sha) (vb : Values α db shb) (different : a ≠ b) :
    (s.setReg a da sha va).setReg b db shb vb =
      (s.setReg b db shb vb).setReg a da sha va := by
  cases s
  unfold State.setReg
  congr 1
  funext d sh n
  by_cases ha : n = a
  · subst n
    simp [different]
  · by_cases hb : n = b
    · subst n
      simp [different, Ne.symm different]
    · simp [ha, hb]

def tail (p : Parameters) := (EarlyAddress.source p).surfaceBody.drop 13

theorem prepare_eq (M : Algebra α) (p : Parameters) (s t : State α)
    (a : Accumulator α p.dimensions) (h : Ready p s t a) :
    run M ((tail p).take 3) t = run M ((suffix p).take 3) t := by
  simp [tail,EarlyAddress.source,EarlyAddress.typed,suffix,source,typed,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat,h.batch,h.head,h.dimensions,h.output,h.normalizer]
  rw [setReg_comm _ "acc" "off_o" _ _ _ _ _ _ (by decide)]
  rw [setReg_comm _ "acc" "out_ptrs" _ _ _ _ _ _ (by decide)]

theorem suffix_eq (M : Algebra α) (p : Parameters) (s t : State α)
    (a : Accumulator α p.dimensions) (h : Ready p s t a) :
    run M (tail p) t = run M (suffix p) t := by
  have sameStore : (tail p).drop 3 = (suffix p).drop 3 := by rfl
  conv_lhs => rw [← List.take_append_drop 3 (tail p)]
  conv_rhs => rw [← List.take_append_drop 3 (suffix p)]
  simp only [run_append,prepare_eq M p s t a h,sameStore]

/-- The complete original source prefix and loop are unchanged. Their proved
Ready invariant justifies independent output-address and division scheduling. -/
theorem execution_eq (M : Algebra α) (p : Parameters) (s : State α) (hN : 0 < p.tokens) :
    FP.Structural.exec M (EarlyAddress.source p) s = FP.Structural.exec M (source p) s := by
  have split : (EarlyAddress.source p).surfaceBody = prelude p ++
      [.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "cur_batch_seq_len")
        (.constNat p.tokens) (loopBody p)] ++ tail p := by rfl
  obtain ⟨a,ha,har⟩ := prefix_run M p s
  obtain ⟨b,hb,hbr⟩ := loop_run M p s a 0 (sequenceLength p s) p.tokens _ har hN
  change run M (EarlyAddress.source p).surfaceBody s = run M (source p).surfaceBody s
  rw [split,source_split]
  simp only [run_append,ha,Option.bind_some,run,step,evalOp_unfold,har.seqlen,Option.bind,hb]
  simpa only [bind,pure,Option.bind,hb] using suffix_eq M p s b _ hbr

/-- Scheduling changes only pure register work before the same final store. -/
theorem tail_safety_iff (M : Algebra α) (p : Parameters) (s t : State α)
    (a : Accumulator α p.dimensions) (h : Ready p s t a) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (tail p) t ↔ Safety.traceSafeList M bounds (suffix p) t := by
  simp [tail,EarlyAddress.source,EarlyAddress.typed,suffix,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,FP.Structural.bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,h.batch,h.head,h.dimensions,h.output,h.normalizer]
  intro _
  constructor <;> intro _ <;> split <;> trivial

theorem safety_iff (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hN : 0 < p.tokens) :
    TraceSafe M bounds (EarlyAddress.source p) s ↔ TraceSafe M bounds (source p) s := by
  obtain ⟨a,ha,har⟩ := prefix_run M p s
  obtain ⟨b,hb,hbr⟩ := loop_run M p s a 0 (sequenceLength p s) p.tokens _ har hN
  have beforeRun : run M ((source p).surfaceBody.take 13) s = some b := by
    have split : (source p).surfaceBody.take 13 = prelude p ++
        [.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "cur_batch_seq_len")
          (.constNat p.tokens) (loopBody p)] := by rfl
    rw [split]
    simp only [run_append,ha,Option.bind_some,run,step,evalOp_unfold,har.seqlen,Option.bind]
    simp only [bind,pure,Option.bind,hb]
  have samePrefix : (EarlyAddress.source p).surfaceBody.take 13 =
      (source p).surfaceBody.take 13 := by rfl
  unfold TraceSafe
  conv_lhs => rw [← List.take_append_drop 13 (EarlyAddress.source p).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 13 (source p).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  simp only [samePrefix]
  constructor
  · rintro ⟨safe,rest⟩
    refine ⟨safe,?_⟩
    intro t ran
    have eq : b=t := Option.some.inj (beforeRun.symm.trans ran)
    subst t
    exact (tail_safety_iff M p s b _ hbr bounds).mp (rest b beforeRun)
  · rintro ⟨safe,rest⟩
    refine ⟨safe,?_⟩
    intro t ran
    have eq : b=t := Option.some.inj (beforeRun.symm.trans ran)
    subst t
    exact (tail_safety_iff M p s b _ hbr bounds).mpr (rest b beforeRun)

#axiomsClean safety_iff
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressSchedule
