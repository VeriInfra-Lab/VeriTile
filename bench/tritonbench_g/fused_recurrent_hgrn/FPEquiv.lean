import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Pair

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Full forward/backward pair. Constexpr T*D is folded before forward pointer
formation; the recurrent instruction graph, reverse pass and all optional
branches retain their source precision and order. Both complete grids, all
outputs, source safety, successful execution and exterior frames are preserved. -/
specification stride_folded_equiv {α : Type} [Inhabited α] (p : Ops α) (observe : α → α)
    (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) :
    Pair.original p observe BH T D BDF BDB U S V ≡[[]]
      Pair.strideFolded p observe BH T D BDF BDB U S V :=
  ProgramIO.Implementation.of_equivalent (Pair.variant p observe BH T D BDF BDB U S V)
    (Pair.equivalent p observe BH T D BDF BDB hF hB U S V)

#axiomsClean stride_folded_equiv
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
