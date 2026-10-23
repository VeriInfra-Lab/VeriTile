import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.IOBounds
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.IOExecution
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeSafety
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeWeightSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeIOSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph IOBounds
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem forward (M : Algebra Term) (h : Call) (epsilon : Term) (bounds : RegionName → Nat)
    (cap : Capacity h Bool.false bounds) (pid : Fin h.rows) (s : State Term) :
    (NativeIO.forwardJob M h epsilon pid.val).Safe bounds s := by
  change TraceSafe (engine M) bounds (Kernels.forwardRuntime h)
    (RuntimeForward.Source.bind (IOExecution.nativeBase pid.val h.rows s) epsilon)
  apply NativeSafety.forward
  · intro j hj;exact (Native.offset_bound h _ _ pid.isLt hj).trans_le cap.x
  · intro j hj;exact hj.trans_le cap.w
  · intro j hj;exact (Native.offset_bound h _ _ pid.isLt hj).trans_le cap.row
  · exact pid.isLt.trans_le cap.r

theorem backward (M : Algebra Term) (h : Call) (bounds : RegionName → Nat)
    (cap : Capacity h Bool.true bounds) (pid : Fin h.rows) (s : State Term) :
    (NativeIO.backwardJob M h Bool.false pid.val h.rows).Safe bounds s := by
  apply NativeSafety.backward
  · intro j hj;exact (Native.offset_bound h _ _ pid.isLt hj).trans_le cap.row
  · intro j hj;exact (Native.offset_bound h _ _ pid.isLt hj).trans_le cap.x
  · intro j hj;exact hj.trans_le cap.w
  · exact pid.isLt.trans_le cap.r

theorem weight (M : Algebra Term) (h : Call) (bounds : RegionName → Nat)
    (cap : Capacity h Bool.true bounds) (pid : Fin (Geometry.featureCount h)) (s : State Term) :
    (NativeIO.backwardJob M h Bool.true pid.val (Geometry.featureCount h)).Safe bounds s := by
  apply NativeWeightSafety.complete
  refine ⟨?_,?_,?_,?_⟩
  · intro row hr j hj;exact (Native.offset_bound h row _ hr hj).trans_le cap.x
  · intro row hr j hj;exact (Native.offset_bound h row _ hr hj).trans_le cap.row
  · intro row hr;exact hr.trans_le cap.r
  · intro j hj;exact hj.trans_le (cap.dw rfl)

theorem map_safe {ι : Type} (jobs : List ι) (job : ι → Invocation Term)
    (bounds : RegionName → Nat) (safe : ∀ i s,(job i).Safe bounds s) (s : State Term) :
    ScheduleSafe bounds (jobs.map job) s := by
  induction jobs generalizing s with
  | nil=>trivial
  | cons i rest ih=>exact ⟨safe i s,fun t _=>ih t⟩

theorem complete (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (bounds : RegionName → Nat) (cap : Capacity h backward bounds) (s : State Term) :
    ScheduleSafe bounds (NativeIO.schedule R h backward epsilon) s := by
  cases backward
  · exact map_safe _ _ bounds (forward (algebra R .forward) h epsilon bounds cap) s
  · unfold NativeIO.schedule
    simp only [if_true]
    split
    · trivial
    · rw [scheduleSafe_append]
      exact ⟨map_safe _ _ bounds (weight (algebra R .weight) h bounds cap) s,
        fun t _=>map_safe _ _ bounds (NativeIOSafety.backward (algebra R .input) h bounds cap) t⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeIOSafety
