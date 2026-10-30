import bench.tritonbench_g.batched_vecmat_mult.Proofs.PairIO

/-! Actual original/block-first full-grid pair on arbitrary source-graph
inputs. The observer retains the existing explicit contraction pass and strict
FP32 evaluation. A contracted carry cannot be a product, so the same product
is fused on both sides; separate addition uses the shared strict add_comm law.
No finite-value restriction or experimental numerical rule is needed. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Equivalence
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open scoped VeriTile.Spec
noncomputable section

abbrev original (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) :=
  ProgramIO.Implementation.original (PairIO.io cfg rest profile p) (fun _ => True)

abbrev blockFirst (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate cfg rest profile p) (fun _ => True)

/-- Both actual complete schedules succeed safely, all observed tagged output
cells agree, and both exterior memory frames hold. Arbitrary expression inputs,
source-dependent reduction trees, every floor-grid CTA and K=0 are retained. -/
specification block_first_strict (cfg : FP.Config) (rest : Algebra Expr)
    (profile : KernelProfile.Selector) (p : Parameters)
    (separate : SourceGraph.InputsSeparate p) :
    original cfg rest profile p ≡[[]] blockFirst cfg rest profile p := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate cfg rest profile p)
    (PairIO.equivalent cfg rest profile p separate)

#axiomsClean block_first_strict
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Equivalence
