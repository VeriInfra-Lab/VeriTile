import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOReadBinding
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOLoopSafety

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

theorem logical_actual_bounds (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (bounds : RegionBounds)
    (hb : (io g r extent M dots).InBounds bounds xs) (idx : GridIndex (Host.grid g.arguments)) :
    IOSourceSafety.Bounds (Host.layout g.arguments) r (logicalInputs g r extent xs idx) bounds := by
  have localBounds := source_bounds g r extent xs valid idx
  have capacity := input_capacity g r extent M dots xs bounds hb
  refine ⟨Nat.lt_of_lt_of_le localBounds.start (capacity r.start (by simp)),
    Nat.lt_of_lt_of_le localBounds.prompt (capacity r.prompt (by simp)),
    Nat.lt_of_lt_of_le localBounds.sequence (capacity r.sequence (by simp)),
    Nat.lt_of_lt_of_le localBounds.request (capacity r.request (by simp)),?_,?_,?_,?_,?_⟩
  · intro i d active dim
    exact Nat.lt_of_lt_of_le (localBounds.query i d active dim) (capacity r.query (by simp))
  · intro cur nonnegative aligned live j active
    exact Nat.lt_of_lt_of_le (localBounds.tokens cur nonnegative aligned live j active) (capacity r.tokens (by simp))
  · intro cur nonnegative aligned live j d active dim
    exact Nat.lt_of_lt_of_le (localBounds.key cur nonnegative aligned live j d active dim) (capacity r.key (by simp))
  · intro cur nonnegative aligned live j d active dim
    exact Nat.lt_of_lt_of_le (localBounds.value cur nonnegative aligned live j d active dim) (capacity r.value (by simp))
  · intro i d active dim
    have backed := hb.2.2.1 (0 : Fin 1) ((coordinate g).symm (idx,(i,d,PUnit.unit)),PUnit.unit)
    simp only [io,memory,address,KernelIO.active,Equiv.apply_symm_apply] at backed
    exact backed ⟨active,dim⟩

theorem actual_input_bounds (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (bounds : RegionBounds)
    (hb : (io g r extent M dots).InBounds bounds xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments)) :
    IOSourceSafety.Bounds (Host.layout g.arguments) r
      (FPReference.inputs (Host.layout g.arguments) r ((invocation g r M dots idx).initial s)) bounds := by
  have agree := input_agreement g r extent M dots xs valid s hin idx
  have localBounds := source_bounds g r extent xs valid idx
  have actualBounds := logical_actual_bounds g r extent M dots xs valid bounds hb idx
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simpa only [agree.batch] using actualBounds.start
  · simpa only [agree.batch] using actualBounds.prompt
  · simpa only [agree.batch] using actualBounds.sequence
  · simpa only [agree.batch] using actualBounds.request
  · intro i d active dim
    simp only [agree.row_eq,agree.length] at active
    simpa only [agree.start,agree.row_eq,agree.head] using actualBounds.query i d active dim
  · intro cur nonnegative aligned live j active
    simp only [agree.mask_eq,agree.end_eq] at live active
    simpa only [agree.request] using actualBounds.tokens cur nonnegative aligned live j active
  · intro cur nonnegative aligned live j d active dim
    simp only [agree.mask_eq,agree.end_eq] at live active
    simpa only [agree.kvHead,
      IOReferenceAgreement.page_eq _ r extent _ _ agree localBounds cur nonnegative aligned live j] using
      actualBounds.key cur nonnegative aligned live j d active dim
  · intro cur nonnegative aligned live j d active dim
    simp only [agree.mask_eq,agree.end_eq] at live active
    simpa only [agree.kvHead,
      IOReferenceAgreement.page_eq _ r extent _ _ agree localBounds cur nonnegative aligned live j] using
      actualBounds.value cur nonnegative aligned live j d active dim
  · intro i d active dim
    simp only [agree.row_eq,agree.length] at active
    simpa only [agree.start,agree.row_eq,agree.head] using actualBounds.output i d active dim

theorem invocation_safe (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (bounds : RegionBounds)
    (hb : (io g r extent M dots).InBounds bounds xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments)) :
    (invocation g r M dots idx).Safe bounds s :=
  IOSourceSafety.kernel_safe _ r _ M dots (Host.block_positive g.arguments) _ bounds
    (actual_input_bounds g r extent M dots xs valid bounds hb s hin idx)

#axiomsClean invocation_safe
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
