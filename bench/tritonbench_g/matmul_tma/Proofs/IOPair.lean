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
    (fun a b i => outputValue M f.half
      (D.apply none {} (batch := []) a b i)) (fun _ _ _ => rfl)
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := original values trivial bounds hb s hin
  obtain ⟨other,candidateRun,sameMem⟩ := SourcePair.source_exec rest I D (params h f) f.half _ final
    (by simpa only [runSchedule_singleton] using ran)
  have safeSame : ∀ initial, (PublicIO.invocation M D (params h f) f.half).Safe bounds initial ↔
      ({ PublicIO.invocation M D (params h f) f.half with kernel := kernelDirectOutput h f } : Invocation Word).Safe bounds initial := by
    intro initial
    exact SourcePair.source_safe M D (params h f) f.half _ bounds
  refine ⟨safe,?_,final,other,ran,?_,?_,frame,?_⟩
  · apply (scheduleSafe_singleton bounds _ s).mpr
    exact (safeSame s).mp ((scheduleSafe_singleton bounds _ s).mp safe)
  · simpa only [variant,runSchedule_singleton] using candidateRun
  · intro o j active
    change mapCell id (final.mem _ _) = mapCell id (other.mem _ _)
    rw [sameMem]
  · intro r offset outside
    rw [sameMem]
    exact frame r offset outside
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.IOPair
