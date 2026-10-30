import bench.tritonbench_g.quant_transpose_kernel.IO
import bench.tritonbench_g.quant_transpose_kernel.Proofs.PublicIO
import VeriTile.Triton.Float.IntegerOutputIO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.ByteIO
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

theorem observes [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (separated : KernelGrid.Separate p) (version : Bool) :
    (IO.grid algebra round p version).ObservesInteger (IO.byteView (IO.grid algebra round p version))
      (IO.formula algebra round p version
        (fun x inverse => ((KernelResult.quantized algebra round x inverse).toNat : Int))) := by
  have implemented := PublicIO.implements algebra round p separated version
  have observed := IntegerProgramIO.ObservesInteger.of_implements implemented
    (IO.byteView (IO.grid algebra round p version))
  have same : (fun vals => (IO.grid algebra round p version).toMemoryIO.mapIntegerOutputs
      (IO.byteView (IO.grid algebra round p version))
      (IO.formula algebra round p version
        (fun x inverse => (KernelResult.quantized algebra round x inverse).toInt) vals)) =
      IO.formula algebra round p version
        (fun x inverse => ((KernelResult.quantized algebra round x inverse).toNat : Int)) := by
    funext vals o j
    fin_cases o
    change ((BitVec.ofInt 8 (KernelResult.quantized algebra round (vals (0 : Fin 2) j) (vals (1 : Fin 2) PUnit.unit)).toInt).toNat : Int) = _
    rw [BitVec.ofInt_toInt]
    rfl
  rw [same] at observed
  exact observed

theorem real (rest : Algebra ℝ) (p : Parameters) (separated : KernelGrid.Separate p)
    (version : Bool) :
    (IO.real rest p version).ObservesInteger (IO.byteView (IO.real rest p version))
      (Reference.realTarget rest p version) := by
  simpa only [IO.real,Reference.realTarget,real_formula] using
    observes (realAlgebra rest) realRound p separated version

theorem equivalent [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (separated : KernelGrid.Separate p) :
    (IO.grid algebra round p Bool.false).IntegerObservedEquivalent (IO.commuted algebra round p)
      (IO.byteView (IO.grid algebra round p Bool.false)) :=
  IntegerProgramIO.IntegerObservedEquivalentOn.of_equivalentOn
    (PublicIO.equivalent algebra round p separated) _

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.ByteIO
