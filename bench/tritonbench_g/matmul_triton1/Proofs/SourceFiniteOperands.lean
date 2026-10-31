import bench.tritonbench_g.matmul_triton1.Proofs.SourceBodyExecution
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteSource

/-! Connect the actual source loop's loaded tiles to modular uint64 addresses.
Only active addresses need allocation bounds; padded pointers remain modular.
There is no second whole-kernel executor. -/
namespace VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
open VeriTile.Triton FP.Structural Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem finite_loaded_tiles (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (initial current : State (Word α))
    (pid : initial.pids 0 < FiniteIndex.count p) (k : Nat)
    (active : k < FiniteIndex.ceil p.innerSize p.blockInner)
    (h : RuntimeInvariant p initial k current) :
    (bodyExecution rest P p initial current k h).val.regs .real [p.blockRows,p.blockInner] "x_sub" =
      some (FiniteOperands.loadA (symbolicPrimitives α) p Bool.false initial (FiniteIndex.word k)) ∧
    (bodyExecution rest P p initial current k h).val.regs .real [p.blockInner,p.blockColumns] "y_sub" =
      some (FiniteOperands.loadB (symbolicPrimitives α) p Bool.false initial (FiniteIndex.word k)) := by
  have operands := FiniteOperands.loads (symbolicPrimitives α) p ready storage Bool.false initial pid k active
  have left := operands.1.trans (FiniteSource.a_eq (symbolicPrimitives α) p canonical initial k)
  have right := operands.2.trans (FiniteSource.b_eq (symbolicPrimitives α) p canonical initial k)
  constructor
  · simpa [bodyExecution,bodyResult] using congrArg some left.symm
  · simpa [bodyExecution,bodyResult] using congrArg some right.symm

#axiomsClean finite_loaded_tiles
end
end VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
