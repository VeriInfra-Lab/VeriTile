import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.PairIO

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Equivalence
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
open scoped VeriTile.Spec
noncomputable section

abbrev original (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) :=
  ProgramIO.Implementation.original (IO.nativeIO isa route scale c order) (fun _ => True)
abbrev tailBoundCSE (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate isa route scale c order) (fun _ => True)

/-- Two actual full FP32 chunk-gradient programs: expose the common absolute
chunk end in the final gate mask. All floating operation provenance, arbitrary
native routing, full inputs, three outputs, safety and frames are retained. -/
specification tail_bound_equiv (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) (legal : order.Perm (allCTAs c)) :
    original isa route scale c order ≡[[]] tailBoundCSE isa route scale c order := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate isa route scale c order)
    (PairIO.equivalent isa route scale c order legal)

#axiomsClean tail_bound_equiv
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Equivalence
