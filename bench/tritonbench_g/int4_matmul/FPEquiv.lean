import bench.tritonbench_g.int4_matmul.Proofs.FoldStrideIO

namespace VeriTile.Bench.Repairs.Int4Matmul.FPEquiv
open VeriTile Triton FP.Structural KernelExecution
open scoped VeriTile.Spec
noncomputable section

/-- Precombine constexpr BK*S in all three dynamic loop-index offsets.
The full packed INT4 source, original FP32 fixed-K8 operations and independent
per-address split orders are unchanged. Only exact discrete arithmetic changes. -/
specification fold_stride_equiv {α : Type} [Inhabited α]
    (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile) (member : t∈Host.sourceTiles)
    (o : Ops α) (rest : Algebra α) (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t)) :
    FoldStrideIO.original x t o rest lanes ≡[[]] FoldStrideIO.optimized x t o rest lanes :=
  PerAddressProgramIO.Implementation.of_equivalentOnCells (FoldStrideIO.variant x t o rest lanes)
    (FoldStrideIO.equivalent x geometry t member o rest lanes complete)

#axiomsClean fold_stride_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int4Matmul.FPEquiv
