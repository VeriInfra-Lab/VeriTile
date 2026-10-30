import bench.repairs.tritonbench_g.context_attn_bloom.IO
import bench.repairs.tritonbench_g.context_attn_bloom.NativeIO
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.RawInterface

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalConnection
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

/-- The public symbolic grid is exactly the regional executor already used
by the retained complete byte/Tensor transport. This does not manufacture a
new flat executor or discard any cells in the caller's original state. -/
theorem native_schedule {α : Type} [Inhabited α]
    (g : Raw.Geometry) (r : Regions) (u : Math.Inputs) (M : Algebra α) (dots : FPSuccess.Dot α)
    (jobs : List (GridIndex (Host.grid g.arguments))) (s : State α) :
    runSchedule (jobs.map (KernelIO.invocation g r M dots)) s=
      FPSuccess.runGrid (Raw.call g r u) M dots jobs s := by
  induction jobs generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    change (exec ((FPSuccess.withDot M dots).withDefaultPrecision .fp32)
      (kernel (Host.layout g.arguments) r (Host.scale g.arguments)) (FPSuccess.atIndex s idx)).bind
        (runSchedule (rest.map (KernelIO.invocation g r M dots)))=_
    simp only [FPSuccess.runGrid]
    congr 1
    funext t
    exact ih t

/-- Every witness of the shared Algorithm IO uses the original serial Real
source runner, including its retained local registers and passed scale. The
existing flat-memory theorem transports this same regional execution. -/
theorem real_execution (g : Raw.Geometry) (r : Regions)
    (jobs : List (GridIndex (Host.grid g.arguments))) (s t : BlockState)
    (trace : AlgorithmExecution
      (jobs.map (fun idx=>.single (RealKernelIO.source g r) (RealKernelIO.launch g idx))) s t) :
    RealStream.run g.arguments r jobs s=some t := by
  induction jobs generalizing s t with
  | nil=>cases trace; rfl
  | cons idx rest ih=>
    cases trace with
    | @cons _ _ _ middle _ ran tail=>
      have he : exec (RealStream.raw g.arguments r) (s.withGridIndex idx)=some middle :=
        (AlgorithmStage.single_runs _ _ _ _).mp ran
      rw [RealStream.run,he]
      exact ih middle t tail

#axiomsClean native_schedule
#axiomsClean real_execution
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalConnection
