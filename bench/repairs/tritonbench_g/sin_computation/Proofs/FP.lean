import bench.repairs.tritonbench_g.sin_computation.Proofs.Host
import bench.repairs.tritonbench_g.sin_kernel.Proofs.FP
import bench.tritonbench_g.sin_computation.Proofs.Source

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.FPProof
open VeriTile.Triton
open Binding (Word)
noncomputable section

abbrev Numerics := SinComputationNative.Numerics
abbrev Target := FP.Structural.Algebra (BitVec 32)
abbrev FP32Storage {M : Type} := @VeriTile.Bench.Repairs.SinKernel.FPProof.FP32Storage M

def scalar (native : Numerics) (x : Word) : Word := ⟨native.sin32 x.bits⟩
/-- Same independently indexed raw-word FP32 sine port as the accepted repair.
It is not an executable correctly-rounded sine or a Real accuracy theorem. -/
def reference (target : Target) (x : Word) : Word := ⟨target.unary (some .fp32) .sin x.bits⟩

/-- Retained numerical scope: every raw FP32 word, including nonfinite values
and signed zero. No execution, output, indexing or memory law is assumed. -/
def SineObligation (native : Numerics) (target : Target) : Prop :=
  ∀ x : BitVec 32, native.sin32 x = target.unary (some .fp32) .sin x

theorem scalar_reference (native : Numerics) (target : Target)
    (h : SineObligation native target) (x : Word) : scalar native x = reference target x :=
  congrArg (fun bits => (⟨bits⟩ : Word)) (h x.bits)

theorem scalar_from_source (native : Numerics) (n : Nat) (x : BitVec 32) :
    (SinComputationNative.compiledScalar ⟨.fp32,.fp32⟩ n).eval native (x.setWidth 64) =
      some ((native.sin32 x).setWidth 64) := by
  rw [SinComputationNative.compiled_scalar]
  simp [SinComputationNative.source,SinComputationNative.Expr.eval,
    SinComputationNative.sine,SinComputationNative.convert,BitVec.setWidth_setWidth_of_le]

theorem same_source (n : Nat) : (fp32Kernel n).eraseDType.toAlgKernel =
    (sourceKernel "x_ptr" "output_ptr" n 4).toAlgKernel :=
  VeriTile.Bench.TritonBenchG.SinComputation.typed_projection .fp32 .fp32 Bool.false _ _ _ _

#print SineObligation
#axiomsClean scalar_reference
#axiomsClean scalar_from_source
#axiomsClean same_source
#specNonCircular reference avoiding [Binding.runWrapper,Native.runJobs]
#specNonCircular SineObligation avoiding [Binding.runWrapper,Native.runJobs]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.FPProof
