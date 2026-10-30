import bench.repairs.tritonbench_g.dropout_triton.Proofs.Source
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.DropoutTriton.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
open VeriTile.Bench.TritonBenchG.DropoutTriton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Full typed-source execution for every primitive backend and all input
values. Eager division remains present even on a false keep branch. -/
def originalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (p : ℝ) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (kernel n p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [kernel, sourceKernel, VeriTile.Triton.DSL.TypedStores.fp32CallValue, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numericLt, numericEq, natLt, bop, store, Region.cast,
    numeric, ofFloat, toFloat, ComputeDType.eraseDType]
  rfl

/-- Reuse the existing integer index rewrite theorem. No floating expression
is transformed, so no numerical law or finite-operand domain is required. -/
theorem execution_equal {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (p : ℝ) (s : State α) :
    FP.Structural.exec (engine M) (kernel n p) s =
      FP.Structural.exec (engine M) (indexCommutedKernel n p) s := by
  let before := (kernel n p).surfaceBody.take 1
  let after := (kernel n p).surfaceBody.drop 2
  have h := FP.ProfiledRewrite.contextual_assign_op M s before after "block_start"
    (.mul .nat .nil (.ref .nat [] "pid") (.constNat blockSize))
    (.mul .nat .nil (.constNat blockSize) (.ref .nat [] "pid"))
    (VeriTile.Bench.TritonBenchG.CacheTransformTriton.IndexRewrite.index_product_commutes
      M s before (.ref .nat [] "pid") (.constNat blockSize))
  exact h

/-- The source writes only active output cells; no value-domain restriction is
needed for this exterior frame, including p=1 and nonfinite operands. -/
theorem original_frame {α : Type} [Inhabited α] (M : Algebra α)
    (n : Nat) (p : ℝ) (s : State α) (r : RegionName) (address : Nat)
    (outside : r ≠ "out" ∨ ∀ i : TileIndex [blockSize],
      s.pids 0 * blockSize + i.1.val < n → address ≠ s.pids 0 * blockSize + i.1.val) :
    (originalExecution M n p s).val.mem r address = s.mem r address := by
  simp only [originalExecution]
  apply State.masked_scatter_frame
  exact outside.imp id (fun h i _ hi => h i hi)

#axiomsClean originalExecution
#axiomsClean execution_equal
#axiomsClean original_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Rewrite
