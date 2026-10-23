import bench.repairs.tritonbench_g.matmul_triton1.Proofs.References
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Bytes

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- The original selected wrapper's guards admit this exact input domain. -/
theorem wrapper_admitted (x : Launch) :
    SharedContract.checkInputs Bool.true x.defaultFP32 (.tensor x.a.metadata) (.tensor x.b.metadata) = .ok () :=
  (SharedContract.checkInputs_iff _ _ _ _).mpr x.admitted

theorem empty_grid {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (x : Launch) (s : State α) (empty : rows x.a.metadata=0 ∨ columns x.b.metadata=0) :
    grid rest P x s = some s ∧ FiniteSource.grid P x.parameters s = some s := by
  have count : Geometry.count (Core.geometry x.parameters) = 0 := by
    rcases empty with h | h <;>
      simp [Geometry.count,Geometry.gridRows,Geometry.gridColumns,Core.geometry,Launch.parameters,h]
  simp [grid,FiniteSource.grid,count,GridExecution.sequence]

theorem empty_post {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) (empty : rows x.a.metadata=0 ∨ columns x.b.metadata=0) :
    image x read mem = read := by
  apply image_empty
  rw [output_numel]
  rcases empty with h | h <;> simp [h]

theorem zero_inner_precision {α : Type} [Inhabited α] (P : Core.Primitives α)
    (x : Launch) (read : Memory Word) (decode : Word → α) (i : Index x)
    (zero : columns x.a.metadata=0) : precisionTarget P x read decode i = P.zero := by
  simp [precisionTarget,LogicalInputs.precisionReference,FiniteIndex.ceil,zero,LogicalInputs.precisionAccum]

theorem zero_inner_real (x : Launch) (read : Memory Word) (decode : Word → ℝ)
    (i : Index x) (zero : columns x.a.metadata=0) : realTarget x read decode i = 0 := by
  simp [realTarget,LogicalInputs.realExpected,zero,RealSum.sum]

#axiomsClean wrapper_admitted
#axiomsClean empty_grid
#axiomsClean empty_post
#axiomsClean zero_inner_precision
#axiomsClean zero_inner_real
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
