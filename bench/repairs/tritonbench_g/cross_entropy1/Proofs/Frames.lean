import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

theorem Mutation.refl {α σ : Type} (obs : Observations α σ) (s : World α σ) (t : Tensor) :
    Mutation obs s s t := ⟨fun _ h => h,fun _ _ _ _ => rfl,fun _ _ _ _ => rfl⟩

theorem Preserves.mutation {α σ : Type} {obs : Observations α σ} {a b : World α σ}
    (h : Preserves obs a b) (dst : Tensor) : Mutation obs a b dst :=
  ⟨h.live,fun r i live _ => h.memory r i live,fun t i live _ => h.integers t i live⟩

theorem Mutation.trans {α σ : Type} {obs : Observations α σ} {a b c : World α σ} {dst : Tensor}
    (ab : Mutation obs a b dst) (bc : Mutation obs b c dst) : Mutation obs a c dst :=
  ⟨fun r h => bc.live r (ab.live r h),
   fun r i live outside => (bc.memory r i (ab.live r live) outside).trans (ab.memory r i live outside),
   fun t i live separate => (bc.integers t i (ab.live _ live) separate).trans (ab.integers t i live separate)⟩

theorem Mutation.prepend {α σ : Type} {obs : Observations α σ} {a b c : World α σ} {dst : Tensor}
    (bc : Mutation obs b c dst) (ab : Preserves obs a b) : Mutation obs a c dst := (ab.mutation dst).trans bc

theorem Mutation.append {α σ : Type} {obs : Observations α σ} {a b c : World α σ} {dst : Tensor}
    (ab : Mutation obs a b dst) (bc : Preserves obs b c) : Mutation obs a c dst := ab.trans (bc.mutation dst)

/-- A mutation of a newly allocated scratch tensor preserves every region
that existed before allocation, including every coordinate outside the views. -/
theorem Preserves.freshMutation {α σ : Type} {obs : Observations α σ} {a b c : World α σ} {dst : Tensor}
    (ab : Preserves obs a b) (bc : Mutation obs b c dst) (fresh : ¬a.live dst.region) : Preserves obs a c := by
  refine ⟨fun r h => bc.live r (ab.live r h),?_,?_⟩
  · intro r i live
    have separate : r≠dst.region := by intro same; exact fresh (same ▸ live)
    exact (bc.memory r i (ab.live r live) (Or.inl separate)).trans (ab.memory r i live)
  · intro t i live
    have separate : t.region≠dst.region := by intro same; exact fresh (same ▸ live)
    exact (bc.integers t i (ab.live _ live) separate).trans (ab.integers t i live)

#axiomsClean Mutation.trans
#axiomsClean Preserves.freshMutation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
