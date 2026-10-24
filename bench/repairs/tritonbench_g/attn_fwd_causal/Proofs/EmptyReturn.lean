import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Host
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Whole
open VeriTile Triton FP.Structural Source
noncomputable section

theorem empty_programs (p : Parameters)
    (empty : p.batches=0 ∨ p.heads=0 ∨ p.length=0) : programs (grid p)=[] := by
  rcases empty with h|h|h <;> simp [programs,grid,h]

/-- Empty calls execute no source instruction and return exactly the allocated
Tensor descriptor, including its unconstrained actual empty strides. -/
theorem invoke_empty {α : Type} [Inhabited α]
    (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters)
    (abi : InputABI p) (before : State α) (alloc : Allocation p abi before)
    (work : List Program) (legal : (programs (grid p)).Perm work)
    (empty : p.batches=0 ∨ p.heads=0 ∨ p.length=0) :
    invoke M dot p abi before alloc work=some ⟨alloc.state,alloc.output⟩ := by
  have hw : work=[] := by simpa [empty_programs p empty] using legal.symm
  simp [invoke,hw,runPrograms]

theorem no_active_empty (p : Parameters)
    (empty : p.batches=0 ∨ p.heads=0 ∨ p.length=0) :
    ¬∃ b h n,b<p.batches ∧ h<p.heads ∧ n<p.length := by
  rintro ⟨b,h,n,hb,hh,hn⟩
  rcases empty with he|he|he <;> omega

#axiomsClean empty_programs
#axiomsClean invoke_empty
#axiomsClean no_active_empty
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Whole
