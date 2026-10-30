import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOSafety
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOResults
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOScalar

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOCorrect
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
noncomputable section

theorem real_forward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (forwardIO realBackend p).Implements (forwardTarget realBackend p HarmonicIO.realForward) :=
  ProgramIO.Implements.intro
    (fun vals bounds inside s _ => IOSafety.forward realBackend p fit vals bounds inside s)
    (IOResults.forward realBackend p fit HarmonicIO.realForward IOScalar.real_forward)

theorem real_backward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (backwardIO realBackend p).ImplementsOn (prepared p)
      (backwardTarget realBackend p (fun xs gs _ => HarmonicIO.realBackward xs gs)) := by
  intro vals admitted bounds inside s binding
  exact ⟨IOSafety.backward realBackend p fit vals bounds inside s,
    IOResults.backward realBackend p fit (fun xs gs _ => HarmonicIO.realBackward xs gs)
      (prepared p) (IOScalar.real_backward p) vals admitted s binding⟩

theorem native_forward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (forwardIO (forwardBackend selectedProfile) p).Implements
      (forwardTarget (forwardBackend selectedProfile) p nativeForward) :=
  ProgramIO.Implements.intro
    (fun vals bounds inside s _ => IOSafety.forward (forwardBackend selectedProfile) p fit vals bounds inside s)
    (IOResults.forward (forwardBackend selectedProfile) p fit nativeForward IOScalar.native_forward)

theorem native_backward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (backwardIO (backwardBackend selectedProfile) p).Implements
      (backwardTarget (backwardBackend selectedProfile) p nativeBackward) :=
  ProgramIO.Implements.intro
    (fun vals bounds inside s _ => IOSafety.backward (backwardBackend selectedProfile) p fit vals bounds inside s)
    (fun vals s binding => IOResults.backward (backwardBackend selectedProfile) p fit nativeBackward
      (fun _ => True) (fun vals _ => IOScalar.native_backward _ _ _) vals trivial s binding)

#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean native_forward
#axiomsClean native_backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOCorrect
