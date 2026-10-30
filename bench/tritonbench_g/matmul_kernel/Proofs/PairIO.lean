import bench.tritonbench_g.matmul_kernel.Proofs.Pair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulKernel.PairIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

/-- The actual full grid is safe and succeeds on both sides, active complete
output cells agree, and both exact exterior cell frames hold under the original
logical inputs and bounds. No floating law or output-value premise is needed. -/
theorem equivalent (p : Config) (valid : p.Valid) (M : Algebra α) (D : DotAcc α) :
    (PublicIO.io p M D).Equivalent (variant p M D) := by
  intro vals _ bounds hb initial _
  have safe : ScheduleSafe bounds ((PublicIO.io p M D).schedule vals) initial := by
    apply PublicIO.schedule_safe p M D bounds
    · intro r k hr hk
      exact hb.2.1 ⟨0,by simp [PublicIO.io,PublicIO.inputs]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    · intro k n hk hn
      exact hb.2.1 ⟨1,by simp [PublicIO.io,PublicIO.inputs]⟩ (⟨k,hk⟩,⟨n,hn⟩,PUnit.unit) trivial
    · intro r n hr hn
      exact hb.2.2.1 ⟨0,by simp [PublicIO.io,PublicIO.outputs]⟩ (⟨r,hr⟩,⟨n,hn⟩,PUnit.unit) trivial
    · exact fun _ h => h
  obtain ⟨final,ran,_,frame⟩ := launch_correct M D p valid initial
  have ioRun : runSchedule ((PublicIO.io p M D).schedule vals) initial=some final :=
    (PublicIO.schedule_eq p M D _ initial).trans ran
  have ioFrame : (PublicIO.io p M D).Frame vals initial final := by
    intro region offset outside
    apply frame region offset
    by_cases other : region≠"C"
    · exact Or.inl other
    · right
      intro w hw
      have miss := outside.1 ⟨0,by simp [PublicIO.io,PublicIO.outputs]⟩
        (⟨w.1,hw.1⟩,⟨w.2,hw.2⟩,PUnit.unit) trivial
      exact miss.resolve_left (by simpa using other)
  refine ⟨safe,(Pair.schedule_safe_equal p M D _ bounds initial).mp safe,
    final,final,ioRun,(Pair.schedule_equal p M D _ initial).symm.trans ioRun,?_,ioFrame,ioFrame⟩
  intro o j _
  rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulKernel.PairIO
