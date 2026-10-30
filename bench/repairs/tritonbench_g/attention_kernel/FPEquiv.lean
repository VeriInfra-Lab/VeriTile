import bench.repairs.tritonbench_g.attention_kernel.Proofs.Pair

namespace VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
open scoped VeriTile.Spec
noncomputable section

/-- Complete source-grid pair for early query loading/scaling. Both original
FP16/BF16 branches, all raw payloads, arbitrary original strides and caller grid
orders are retained. Only independent prologue statements are rescheduled;
there is no floating numerical rewrite or experimental rule. -/
specification query_early_equiv {α β : Type} [Inhabited β]
    (M : Algebra β) (dot : Dot β) (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (hbm : 0<p.bm) (separate : Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) (encode : α → β)
    (observe : TileIndex (IO.TensorShape p) → β → α) :
    Pair.original M dot p work encode observe ≡[[]]
      Pair.queryEarly M dot p work encode observe :=
  ProgramIO.Implementation.of_equivalent (Pair.variant M dot p work encode observe)
    (Pair.equivalent M dot p bases domain hbm separate work order encode observe)

#axiomsClean query_early_equiv
end
end VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
