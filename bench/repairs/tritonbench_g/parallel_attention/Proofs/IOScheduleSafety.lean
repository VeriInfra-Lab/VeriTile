import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOBounds
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyForwardWhole
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyQueryWhole
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
variable {α : Type} [Inhabited α]

theorem schedule_safe (jobs : List (Invocation α)) (bounds : RegionName → Nat)
    (safe : ∀ job∈jobs,∀ s,job.Safe bounds s) (s : State α) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    exact ⟨safe job (by simp) s,fun t _=>ih (fun j hj=>safe j (by simp [hj])) t⟩

theorem combined_safe (shape : Shape) (scale : ℝ) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO α .backward shape jobs).InBounds bounds xs) (p : Program shape 64)
    (M : Algebra α) (D : DotPrimitive α) (s : State α) :
    TraceSafe (withDot M D) bounds (combinedKernel shape scale p.tile) s := by
  change Safety.traceSafeList _ _
    ((queryKernel (dqConfig shape scale p.tile)).surfaceBody ++
      [.effectMarker "tl.debug_barrier"] ++
      (backwardKernel (backwardConfig shape scale p.tile)).surfaceBody) s
  rw [List.append_assoc,Safety.traceSafeList_append]
  refine ⟨IOSafety.query_source M D _ bounds s (query_bounds shape scale jobs xs bounds h p),?_⟩
  intro t _
  apply Safety.traceSafeList_cons_intro
  · simp [Safety.stepSafe_unfold]
  · intro u _
    exact IOSafety.backward_source M D _ bounds u (backward_bounds shape scale jobs xs bounds h p)

theorem real_safe (direction : Direction) (shape : Shape) (scale : ℝ)
    (xs : ChannelValues ℝ (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO ℝ direction shape (realJobs direction shape scale)).InBounds bounds xs) (s : State ℝ) :
    ScheduleSafe bounds (realJobs direction shape scale) s := by
  apply schedule_safe
  cases direction with
  | forward =>
    intro job member t
    obtain ⟨p,_,rfl⟩:=List.mem_map.mp member
    change TraceSafe _ bounds _ ((forwardJob _ _ shape scale p).initial t)
    rw [forward_initial]
    exact IOSafety.forward_source _ _ _ bounds _ (forward_bounds shape scale _ xs bounds h p t)
  | backward =>
    intro job member t
    obtain ⟨p,_,rfl⟩:=List.mem_map.mp member
    change TraceSafe _ bounds _ ((combinedJob shape scale p).initial t)
    rw [combined_initial]
    exact combined_safe shape scale _ xs bounds h p _ _ _

theorem native_safe (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ)
    (xs : ChannelValues NativeStage.Expr (inputPorts d.shape)) (bounds : RegionName → Nat)
    (h : (dataIO NativeStage.Expr direction d.shape (nativeJobs direction d scale)).InBounds bounds xs)
    (s : State NativeStage.Expr) : ScheduleSafe bounds (nativeJobs direction d scale) s := by
  apply schedule_safe
  cases direction with
  | forward =>
    intro job member t
    obtain ⟨p,_,rfl⟩:=List.mem_map.mp member
    change TraceSafe _ bounds _ ((forwardJob _ _ d.shape scale p).initial t)
    rw [forward_initial]
    exact IOSafety.forward_source _ _ _ bounds _ (forward_bounds d.shape scale _ xs bounds h p t)
  | backward =>
    intro job member t
    obtain ⟨p,_,member⟩:=List.mem_flatMap.mp member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · change TraceSafe _ bounds _ ((queryJob _ _ d.shape scale p).initial t)
      rw [query_initial]
      exact IOSafety.query_source _ _ _ bounds t (query_bounds d.shape scale _ xs bounds h p)
    · change TraceSafe _ bounds _ ((keyValueJob _ _ d.shape scale p).initial t)
      rw [keyValue_initial]
      exact IOSafety.backward_source _ _ _ bounds t (backward_bounds d.shape scale _ xs bounds h p)

#axiomsClean real_safe
#axiomsClean native_safe
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
