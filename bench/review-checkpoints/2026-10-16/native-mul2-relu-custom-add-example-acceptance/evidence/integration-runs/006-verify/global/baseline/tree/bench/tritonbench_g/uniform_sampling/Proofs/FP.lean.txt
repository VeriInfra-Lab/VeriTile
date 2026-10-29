import bench.tritonbench_g.uniform_sampling.Kernels
import VeriTile.Triton.Float.Structural

/-! Complete typed source/candidate equivalence. Only the integer lane-index
multiplication changes. The complete identical suffix keeps the modular
counter addition, all four Philox fields, fp32 uniform and affine operations,
and all four destination conversions/stores. This is structural execution
equality, not a numerical refinement from unfused arithmetic to native FMA.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.FPProof
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def indexStmt (swapped : Bool) (B : Nat) : ComputeStmt :=
  .assign .nat [B] "i4" (.alg (.add .nat .scalarL
    (if swapped then
      .mul .nat .nil (.ref .nat [] "BLOCK_arg") (.programId 0) else
      .mul .nat .nil (.programId 0) (.ref .nat [] "BLOCK_arg"))
    (.arange B)))

theorem source_split (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody =
      (source storage Bool.false y N seed offset lower upper B hB).surfaceBody.take 7 ++
      indexStmt swapped B ::
      (source storage Bool.false y N seed offset lower upper B hB).surfaceBody.drop 8 := by
  cases storage <;> cases swapped <;> rfl

theorem index_step (B : Nat) {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.step M (indexStmt Bool.false B) s =
      FP.Structural.step M (indexStmt Bool.true B) s := by
  cases h : s.regs .nat [] "BLOCK_arg" <;>
    simp [indexStmt, FP.Structural.step, FP.Structural.evalExpr,
      FP.Structural.evalOp_unfold, h, FP.Structural.numeric]
  congr 1
  funext i
  simp [FP.Structural.bop, Nat.mul_comm]

theorem run_insert {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (before after : List ComputeStmt) (a b : ComputeStmt)
    (h : ∀ s, FP.Structural.step M a s = FP.Structural.step M b s)
    (s : FP.Structural.State α) :
    FP.Structural.run M (before ++ a :: after) s =
      FP.Structural.run M (before ++ b :: after) s := by
  rw [FP.Structural.run_append, FP.Structural.run_append]
  cases hp : FP.Structural.run M before s <;>
    simp only [Option.bind_none, Option.bind_some, FP.Structural.run, h]

/-- Every state component and failure result agrees for every floating
algebra. This needs no IEEE rewrite law, range restriction on the bounds,
or successful-execution hypothesis. Original ABI conditions are separate
premises of the successful source-correctness contract. -/
theorem whole_state (storage : Storage) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B)
    {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.exec M (source storage Bool.false y N seed offset lower upper B hB) s =
      FP.Structural.exec M (source storage Bool.true y N seed offset lower upper B hB) s := by
  change FP.Structural.run M (source storage Bool.false y N seed offset lower upper B hB).surfaceBody s =
    FP.Structural.run M (source storage Bool.true y N seed offset lower upper B hB).surfaceBody s
  rw [source_split storage Bool.false, source_split storage Bool.true]
  exact run_insert M _ _ _ _ (index_step B M) s

#axiomsClean source_split
#axiomsClean index_step
#axiomsClean whole_state

end
end VeriTile.Bench.TritonBenchG.UniformSampling.FPProof
