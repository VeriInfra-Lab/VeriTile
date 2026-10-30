import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Body

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

/-- The real source range, with its register carry and row store at every
iteration. The mathematical recurrence is already fixed by Context. -/
theorem loopExec (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (n : Nat) (hn : n≤c.T) (ctx : Context M c initial scale n current) :
    ∃ t, range (engine M) "_i" n c.T 1 (parts c).body current=some t ∧
      Context M c initial scale c.T t := by
  by_cases live : n<c.T
  · let next := bodyExec M c initial current scale n ctx
    obtain ⟨t,ht,hctx⟩ := loopExec M c initial next.val scale (n+1) (by omega)
      (body_context M c initial current scale n ctx live)
    refine ⟨t,?_,hctx⟩
    rw [range]
    simpa only [show (1:Nat)≠0 by decide,live,if_false,if_true,next.property,Option.bind_some] using ht
  · have done : n=c.T := by omega
    subst n
    exact ⟨current,by simp [range],ctx⟩
termination_by c.T-n
decreasing_by omega

/-- Both the empty loop and the final backedge stop before reading any body
pointer register. No hypothesis about those dead pointer values is needed. -/
theorem stopped_range (M : Algebra α) (c : Shape) (current : State α) :
    range (engine M) "_i" c.T c.T 1 (parts c).body current=some current := by
  simp [range]

#axiomsClean loopExec
#axiomsClean stopped_range
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
