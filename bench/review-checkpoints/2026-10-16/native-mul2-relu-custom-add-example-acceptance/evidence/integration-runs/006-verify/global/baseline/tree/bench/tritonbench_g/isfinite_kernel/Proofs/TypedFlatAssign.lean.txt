import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatExpr
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatControl

/-! The immediate expression-to-statement bridge. Assignments translate
all register channels, including block-pointer payloads. Its premise is only
the expression child commutation at the actual incoming state. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

theorem assign_transport (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : ComputeExpr d sh) (s : State α)
    (he : ExprCommutes A M background flatUndef s e) :
    StepTransport A M background flatUndef (.assign d sh name e) s := by
  intro t ht
  rw [flattenComputeStmt, FP.Structural.step]
  change ((evalExpr M (flattenExpr A e) (flattenState A background flatUndef s)).bind
    fun v => some ((flattenState A background flatUndef s).setReg name d sh v)) =
      some (flattenState A background flatUndef t)
  unfold ExprCommutes at he
  rw [he]
  cases hv : evalExpr M e s with
  | none =>
      simp only [FP.Structural.step, hv, bind, Option.bind_none] at ht
      cases ht
  | some v =>
      have hstate : s.setReg name d sh v = t := by
        simpa only [FP.Structural.step, hv, bind, Option.bind_some,
          Option.pure_def, Option.some.injEq] using ht
      subst t
      simp only [Option.map_some, Option.bind_some]
      exact congrArg some (flatten_setReg A background flatUndef s name d sh v).symm

/-- Ordinary expression children may return pointers. Their translated
payloads must be used; data equality is not substituted for OpCommutes. -/
theorem assign_alg_transport (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : Op d sh) (s : State α)
    (he : OpCommutes A M background flatUndef s none e) :
    StepTransport A M background flatUndef (.assign d sh name (.alg e)) s := by
  apply assign_transport A M background flatUndef d sh name (.alg e) s
  unfold ExprCommutes
  change evalOp M none (A.flattenOp e) (flattenState A background flatUndef s) =
    (evalOp M none e s).map (trValues A)
  exact he

theorem assign_alg_data_transport (d : AlgDType) (sh : TileShape) (name : RegName)
    (e : Op d sh) (s : State α)
    (hp : d ≠ .ptr) (hb : d ≠ .blockPtr)
    (he : OpSame A M background flatUndef s none e) :
    StepTransport A M background flatUndef (.assign d sh name (.alg e)) s := by
  exact assign_alg_transport A M background flatUndef d sh name e s
    ((opSame_iff_commutes A M background flatUndef s none e hp hb).mp he)

#axiomsClean assign_transport
#axiomsClean assign_alg_transport
#axiomsClean assign_alg_data_transport

end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
