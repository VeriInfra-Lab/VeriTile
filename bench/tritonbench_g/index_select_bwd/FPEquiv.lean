import bench.tritonbench_g.index_select_bwd.Proofs.Adaptive

namespace VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- The original fixed-width JIT specialization and the tail-aware power-of-two
specialization execute the same FP32 load/add/store graph at every active
column. Both succeed safely and preserve the full exterior frame. No floating
value is excluded, and no numerical rewrite or experiment is used. -/
specification adaptive_tail_equiv {A : Type} [Inhabited A] (rest : Algebra A)
    (add : A → A → A) (p : Launch) (stride : p.stride = 1)
    (count : p.count ≤ p.block) (small : p.count ≤ 512) :
    Interface.original rest add p ≡[[]] Interface.adaptive rest add p :=
  ProgramIO.Implementation.of_equivalent (Interface.adaptiveVariant rest add p)
    (Interface.adaptive_equivalent rest add p stride count small)

#print_fp_assumptions adaptive_tail_equiv
#axiomsClean adaptive_tail_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
