import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.PairOutputAddress

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p (outputInitial M p) s)
    (head : Head p pid s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i_k" 0 (keyBlocks p) 1 (parts p scale).body s=some t ∧
      OutputReady p pid input t ∧ HasOutput p (outputStates M D p pid input (keyBlocks p)) t ∧ Head p pid t := by
  let I := fun n (t : State α) => OutputReady p pid input t ∧
    HasOutput p (outputStates M D p pid input n) t ∧ Head p pid t
  have hs : ∀ n, n<keyBlocks p → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (parts p scale).body (t.setReg "i_k" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t hi
    let u := outputBody M D p scale pid input t n (outputStates M D p pid input n) hi.1 hi.2.1
    have ready := outputBody_ready M D p scale pid input t n _ hi.1 hi.2.1
    refine ⟨u.val,?_,ready.1,ready.2,body_head M D p scale pid input t n _ hi.1 hi.2.1 hi.2.2⟩
    rw [body_run M D p scale pid input t n hi.1 hi.2.2]
    exact u.property
  exact range_invariant (engine (withDot M D)) "i_k" (parts p scale).body (keyBlocks p)
    I hs 0 (Nat.zero_le _) s ⟨ctx,ha,head⟩

#axiomsClean loop

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
