import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.Conv2d.KernelNumeric
open VeriTile.Triton FP.Structural KernelBinding KernelGrid
noncomputable section
variable {α : Type} [Inhabited α]

/-- The shared DSL's complete original grid computes the independent initial
array graph. Neither execution success nor any output/heap equality is a premise. -/
theorem launch_reference (o : Operations α) (s : Shape) (v : Valid s) (f : Flags)
    (t : State (Value α)) :
    ∃ u,KernelGrid.launch (algebra o) (halfDot o) (defaultDot o) s f t=some u ∧
      KernelGrid.SameInputs u t ∧
      (∀ x,s.Active x → u.mem "output" (outputPointer s x)=
        .mk .fp32 (.scalar (reference o s f (memory o t) x))) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ s.size≤a → u.mem r a=t.mem r a) := by
  obtain ⟨u,he,hi,hr,hf⟩:=KernelGrid.launch_correct (algebra o) (halfDot o) (defaultDot o) s v f t
  refine ⟨u,he,hi,?_,hf⟩
  intro x hx
  rw [hr x hx,point_reference o s v f t x hx]

#axiomsClean launch_reference
end
end VeriTile.Bench.Repairs.Conv2d.KernelNumeric
