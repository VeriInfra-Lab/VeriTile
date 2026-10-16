import bench.tritonbench_g.uniform_sampling.Kernels
import VeriTile.Triton.Float.Structural

/-! Successful typed execution of the four original masked stores, including
their source precision annotations and all six destination conversions. The
body is taken directly from the full source. Its independence from the
earlier runtime arguments is checked by source_slice below.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedStores
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def body (storage : Storage) (y : RegionName) (N B : Nat)
    (hB : SourceDomain.LegalBlock B) : List ComputeStmt :=
  (source storage Bool.false y N 0 0 0 0 B hB).surfaceBody.drop 19

theorem source_slice (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 19 =
      body storage y N B hB := by
  cases storage <;> cases swapped <;> rfl

theorem masked_regs {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (N : Nat) (l : List ι) (s : State α) :
    (l.foldl (fun t i => if off i < N then t.write r (off i) (val i) else t) s).regs =
      s.regs := by
  rw [State.masked_scatter_eq_filter, State.scatter_regs]

theorem stores_exec {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (y : RegionName) (N B : Nat) (hB : SourceDomain.LegalBlock B)
    (s : State α) (v0 v1 v2 v3 : Values α .real [B])
    (o0 o1 o2 o3 : TileIndex [B] → Nat)
    (hv0 : s.regs .real [B] "r0" = some v0)
    (hv1 : s.regs .real [B] "r1" = some v1)
    (hv2 : s.regs .real [B] "r2" = some v2)
    (hv3 : s.regs .real [B] "r3" = some v3)
    (ho0 : s.regs .nat [B] "off_0" = some o0)
    (ho1 : s.regs .nat [B] "off_1" = some o1)
    (ho2 : s.regs .nat [B] "off_2" = some o2)
    (ho3 : s.regs .nat [B] "off_3" = some o3) :
    ∃ v, run M (body storage y N B hB) s = some v := by
  cases storage
  all_goals simp [body, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, numericLt, natLt, bop, ofFloat, toFloat,
    Random.i32ArangeSize, Region.cast, masked_regs, hv0, hv1, hv2, hv3, ho0, ho1, ho2, ho3]

#axiomsClean source_slice
#axiomsClean masked_regs
#axiomsClean stores_exec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedStores
