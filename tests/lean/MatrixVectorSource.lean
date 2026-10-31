import bench.tritonbench_g.matrix_vector_multip.Correct
import bench.tritonbench_g.matrix_vector_multip.FPEquiv
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.MatrixVectorSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section

/-- Six ABI bindings and all eighteen Python body statements, on both sides. -/
theorem complete_sources (p : Typed.Parameters) :
    (Kernels.fromParameters p).surfaceBody.length=24 ∧
      (Kernels.fmaFromParameters p).surfaceBody.length=24 := by constructor <;> rfl

/-- The source while body retains the seventh statement, its explicit increment. -/
theorem retained_increment (p : Typed.Parameters) :
    (Source.loopBody p).getLast?=some (.assign .nat [] "m"
      (.alg (.add .nat .nil (.ref .nat [] "m") (.constNat p.blockColumns)))) := by rfl

theorem actual_scalar_wrap {α : Type} [Inhabited α] (A : Algebra α)
    (p : Typed.Parameters) (s : State α) :
    (Source.preludeExecution A {p with rows:=2^64+3} s).val.regs .nat [] "N"=some (fun _=>3) := by
  simpa using Source.prelude_metadata A {p with rows:=2^64+3} s "N" (2^64+3)
    (by simp [Source.scalarParameters])

def unsignedFull : ComputeKernel :=
  with_triton_signature ({} : CallSignature) in triton {
    i = tl.full((), 18446744073709551619, tl.uint64)
  }

theorem full_wrap {α : Type} [Inhabited α] (A : Algebra α) (s : State α) :
    exec A unsignedFull s=some (s.setReg "i" .nat [] (fun _=>3)) := by
  simp [unsignedFull,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Region.cast,FP.Structural.bop,Option.bind]
  rfl

/-- error: affine while bound and stride must remain unchanged -/
#guard_msgs in
example (stride : Nat) : ComputeKernel :=
  with_triton_signature ({scalars:=[("N",.integer)],constexpr:=["stride"]} : CallSignature) in triton {
    N = tl.cast(N, tl.uint64)
    i = tl.full((), 0, tl.uint64)
    while i < N {
      N += stride
      i += stride
    }
  }

#axiomsClean complete_sources
#axiomsClean retained_increment
#axiomsClean actual_scalar_wrap
#axiomsClean full_wrap
end
end VeriTile.Tests.MatrixVectorSource
