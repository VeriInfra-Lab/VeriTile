import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Proofs.IOCorrect
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelNumerics

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPEquiv
open VeriTile Triton FP.Structural
open Kernel Kernel.Scalar
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The complete source schedule with FP32 graph-word input leaves and both
runtime clamp words. The native observer keeps the selected prefix plan. -/
def nativeIO (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool) (ports : Numeric.ScalarPorts)
    (lower upper : BitVec 32) : ProgramIO (BitVec 32) (Term (BitVec 32)) :=
  Interface.programIO (Numeric.nativeModel ports reverseBias) c plan hasBias useSoftplus lower upper

/-- Both independent native recurrences, successful complete-grid execution,
actual source access safety and exact writable-memory frame. The existing
fixed-arity scalar laws are unchanged primitive-model boundaries. -/
theorem kernel_fp32 (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool)
    (target : Numeric.Target) (unary : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (scalar : Numeric.ScalarLaws target unary ports) (lower upper : BitVec 32) :
    nativeIO c plan hasBias reverseBias useSoftplus ports lower upper ⊨
      Interface.nativeTarget (Numeric.referenceModel target unary reverseBias)
        c plan hasBias useSoftplus lower upper := by
  simpa only [nativeIO, Numeric.model_eq target unary ports scalar reverseBias] using
    Interface.implements (Numeric.nativeModel ports reverseBias) c sizes plan hasBias useSoftplus lower upper

#specNonCircular Interface.nativeTarget avoiding [Kernel.launch, Scalar.interpret]
#axiomsClean kernel_fp32
#stmtSurfaceSubset kernel_fp32 ⊆
  [nativeIO, Interface.nativeTarget, Geometry, Geometry.BC, Geometry.BH,
   Numeric.ScalarLaws, Numeric.ScalarPorts, Numeric.Target, Numeric.UnaryModels,
   Numeric.Word, Numeric.referenceModel, PrefixPlan, Term, instInhabitedTerm,
   Sizes, ProgramIO.Implements]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPEquiv
