import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Grid

namespace VeriTile.Bench.Repairs.SoftmaxTriton1

/-- A theorem about a concrete source evaluator. Public Real/FP entries provide
constructed instances proved from the actual AST, never an assumption about a
whole output or launch. -/
structure SourceContract {S C V} (a : Access S C V)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S)
    (formula : Nat → Nat → (Nat → V) → Nat → V) : Prop where
  row : ∀ x y B p s t values, SameShape x y → y.Injective →
    0<B → x.cols≤B → p<x.rows → Inputs a x values s →
    evaluate x y B p s=some t →
    (∀ c, c<x.cols → a.read t y.region (y.index p c)=formula x.cols B (values p) c) ∧
    FrameRows a y [p] s t

end VeriTile.Bench.Repairs.SoftmaxTriton1
