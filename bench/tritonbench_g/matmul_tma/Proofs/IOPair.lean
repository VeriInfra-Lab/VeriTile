import bench.tritonbench_g.matmul_tma.Proofs.SourcePair
import bench.tritonbench_g.matmul_tma.Proofs.PublicIO
import bench.tritonbench_g.matmul_tma.IO

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.IOPair
open VeriTile.Triton FP.Structural FPValue Host StaticChecked.Native
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The generic complete source IO theorem supplies success and both frames;
its floating formula is used only internally, not as a substitute for the pair. -/
theorem equivalent (rest : Algebra Word) (I : Instructions) (profile : Profile) (h : Shape) (f : Flags) :
    (io rest I profile h f).Equivalent (variant rest I profile h f) := by
  let M := StaticChecked.Native.algebra rest I
  let D := sourceDot I profile
  have original := PublicIO.implements M D (params h f) f.half
    (c_injective h f StaticChecked.regions)
    (fun a b i => M.cast none .fp32 (outputDType f.half)
      (D.apply (some .fp32) {} (batch := []) a b i)) (fun _ _ _ => rfl)
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := original values trivial bounds hb s hin
  have same : ∀ initial, (PublicIO.invocation M D (params h f) f.half).run initial =
      ({ PublicIO.invocation M D (params h f) f.half with kernel := kernelDirectOutput h f } : Invocation Word).run initial := by
    intro initial
    exact SourcePair.source_exec rest I profile (params h f) f.half _
  have safeSame : ∀ initial, (PublicIO.invocation M D (params h f) f.half).Safe bounds initial ↔
      ({ PublicIO.invocation M D (params h f) f.half with kernel := kernelDirectOutput h f } : Invocation Word).Safe bounds initial := by
    intro initial
    exact SourcePair.source_safe M D (params h f) f.half _ bounds
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · apply (scheduleSafe_singleton bounds _ s).mpr
    exact (safeSame s).mp ((scheduleSafe_singleton bounds _ s).mp safe)
  · rw [runSchedule_singleton] at ran
    simpa only [variant,runSchedule_singleton] using (same s).symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.IOPair
