import bench.tritonbench_g.fast_rms_layernorm.Proofs.IOBounds
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealRows
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightTrace
import bench.tritonbench_g.fast_rms_layernorm.Proofs.NativeSafety
import bench.tritonbench_g.fast_rms_layernorm.Proofs.NativeWeightSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph IOBounds
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem weight_bounds (h : Call) (bounds : RegionName → Nat) (cap : Capacity h Bool.true bounds) (pid : Nat) :
    RealWeightSafety.Bounds h 256 pid bounds := by
  refine ⟨?_,?_,?_,?_⟩
  · intro row hr j hj
    exact (Native.offset_bound h row _ hr hj).trans_le cap.x
  · intro row hr j hj
    exact (Native.offset_bound h row _ hr hj).trans_le cap.row
  · intro row hr
    exact hr.trans_le cap.r
  · intro j hj
    exact hj.trans_le (cap.dw rfl)

theorem real_row (h : Call) (backward : Bool) (bounds : RegionName → Nat)
    (cap : Capacity h backward bounds) (pid : Fin h.rows) (s : BlockState) :
    (RealIO.job h backward Bool.false pid.val h.rows).Safe bounds s := by
  let u:=(RealIO.job h backward Bool.false pid.val h.rows).initial s
  have p : u.pids 0<h.rows:=pid.isLt
  cases backward
  · apply ForwardRun.trace_safe h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps
    · intro j hj;exact (Native.offset_bound h _ _ p hj).trans_le cap.x
    · intro j hj;cases h.gemma <;> simpa [ForwardRun.weightOffset] using hj.trans_le cap.w
    · intro j hj;exact (Native.offset_bound h _ _ p hj).trans_le cap.row
    · simpa using p.trans_le cap.r
  · apply BackwardRun.trace_safe h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps
    · intro j hj;exact (Native.offset_bound h _ _ p hj).trans_le cap.row
    · intro j hj;exact (Native.offset_bound h _ _ p hj).trans_le cap.x
    · intro j hj;exact hj.trans_le cap.w
    · simpa using p.trans_le cap.r

theorem real_weight (h : Call) (bounds : RegionName → Nat) (cap : Capacity h Bool.true bounds)
    (pid : Fin (Geometry.featureCount h)) (s : BlockState) :
    (RealIO.job h Bool.true Bool.true pid.val (Geometry.featureCount h)).Safe bounds s :=
  RealWeightTrace.complete h 256 bounds _ (weight_bounds h bounds cap pid.val)

theorem real_map {ι : Type} (jobs : List ι) (job : ι → PreparedAlgorithmInvocation)
    (bounds : RegionName → Nat) (safe : ∀ i s,(job i).Safe bounds s) (s : BlockState) :
    PreparedAlgorithmScheduleSafe bounds (jobs.map job) s := by
  induction jobs generalizing s with
  | nil=>trivial
  | cons i rest ih=>exact ⟨safe i s,fun t _=>ih t⟩

theorem real_append (left right : List PreparedAlgorithmInvocation) (bounds : RegionName → Nat)
    (s : BlockState) (first : PreparedAlgorithmScheduleSafe bounds left s)
    (next : ∀ s,PreparedAlgorithmScheduleSafe bounds right s) :
    PreparedAlgorithmScheduleSafe bounds (left++right) s := by
  induction left generalizing s with
  | nil=>exact next s
  | cons job rest ih=>exact ⟨first.1,fun t ran=>ih t (first.2 t ran)⟩

theorem real (h : Call) (backward : Bool) (bounds : RegionName → Nat)
    (cap : Capacity h backward bounds) (s : BlockState) :
    PreparedAlgorithmScheduleSafe bounds (RealIO.schedule h backward) s := by
  cases backward
  · exact real_map _ _ bounds (real_row h Bool.false bounds cap) s
  · unfold RealIO.schedule
    simp only [if_true]
    split
    · trivial
    · apply real_append
      · exact real_map _ _ bounds (real_weight h bounds cap) s
      · exact real_map _ _ bounds (real_row h Bool.true bounds cap)

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOSafety
