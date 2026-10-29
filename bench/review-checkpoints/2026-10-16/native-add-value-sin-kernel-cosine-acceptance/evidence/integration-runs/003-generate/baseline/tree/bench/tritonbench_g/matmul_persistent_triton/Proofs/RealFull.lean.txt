import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealIO
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealVariant

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFull
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000

/-- Both storage profiles project to the proved full real source. -/
theorem algorithm (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :
    (p.source storage v).eraseDType.toAlgKernel = RealSource.kernel p v := by
  simp only [ComputeKernel.toAlgKernel,RealSource.projection]

/-- Original and candidate share complete successful region execution,
independent GEMM outputs, original input preservation and the exact raw frame. -/
theorem executes (p : Parameters) (v : Bool) (h : RealIO.Admissible p)
    (s : BlockState) (xs : RealMath.Inputs) (hp : s.pids 0 < p.SMS)
    (hi : RealInput.Represents p xs s) :
    ∃ t, exec (RealSource.kernel p v) s = some t ∧
      RealCompleted.Holds p xs (s.pids 0) (RealLoopState.work p (s.pids 0)) t ∧
      RealInput.Represents p xs t ∧
      ∀ r o, RealLoop.Outside p (s.pids 0) r o → t.mem r o = s.mem r o := by
  rw [RealVariant.equivalent]
  exact RealContract.executes p xs s h.m h.n h.k h.bm h.bn h.bk h.gm h.sms hp
    h.output_a h.output_b h.output_injective hi

/-- Whole-source safety for both actual variants, under original logical
allocation bounds and original tagged input cells. -/
theorem safe (p : Parameters) (v : Bool) (h : RealIO.Admissible p)
    (bounds : RegionBounds) (s : BlockState) (xs : RealMath.Inputs)
    (hp : s.pids 0 < p.SMS) (hb : RealAccess.Bounds p bounds)
    (hi : RealInput.Represents p xs s) :
    (RealSource.kernel p v).TraceSafe bounds s :=
  (RealVariant.safe_iff p v bounds s).mpr
    (RealSafety.safe p bounds xs s h.m h.n h.k h.bk h.gm h.sms hp h.output_a h.output_b hb hi)

/-- Both complete actual variants implement the finite-flat named contract.
The argument uses actual execution, trace safety and flattening admission. -/
theorem correct (p : Parameters) (v : Bool) (h : RealIO.Admissible p) :
    RealIO.Implements p (RealSource.kernel p v) := by
  intro A s xs hd hr hc hu hl hb hi
  obtain ⟨t,he,hy,hinput,hframe⟩ := executes p v h s xs hl.pid_bound hi
  refine ⟨A.flattenState t,?_,RealIO.results p A xs s t hd hr hb hy hinput hframe⟩
  rw [A.exec_flatten hd hc _ s (safe p v h A.extent s xs hl.pid_bound hb hi)
    (RealPureSafety.flatten_ok p v) hu,he,Option.map_some]

/-- Storage-parametric public real-erasure contract. This claim is exact real
mathematics; the original typed FP equivalence is a separate theorem. -/
theorem storage_correct (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (h : RealIO.Admissible p) :
    RealIO.Implements p (p.source storage v).eraseDType.toAlgKernel := by
  rw [algorithm]
  exact correct p v h

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFull
