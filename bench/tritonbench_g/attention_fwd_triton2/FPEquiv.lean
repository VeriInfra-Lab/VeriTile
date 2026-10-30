import bench.tritonbench_g.attention_fwd_triton2.Proofs.Pair

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.FPEquiv
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
open scoped VeriTile.Spec
noncomputable section

def original {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters) (encode : α → β)
    (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α) (work : List Program) :=
  ProgramIO.Implementation.original (io M dot p encode observe work) (fun _ => True)

def shifted {α β : Type} (M : Algebra β) (dot : Dot β) (p : Parameters) (encode : α → β)
    (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α) (work : List Program) :=
  ProgramIO.Implementation.ofVariant (Pair.variant M dot p encode observe work) (fun _ => True)

/-- Complete fixed-BLOCK_M source pair. The single pre-loop row-base multiply
becomes a shift; source precision, all stages/masks, 3D grid, safety and frames
are retained. There is no floating numerical rewrite or finite-word restriction. -/
specification grid_equiv {α β : Type} [Inhabited α] [Inhabited β]
    (M : Algebra β) (dot : Dot β) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (encode : α → β) (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    original M dot p encode observe work ≡[[]] shifted M dot p encode observe work :=
  ProgramIO.Implementation.of_equivalent (Pair.variant M dot p encode observe work)
    (Pair.equivalent M dot p bases dom sep encode observe work order)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.FPEquiv
