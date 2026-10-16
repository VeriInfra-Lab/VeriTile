import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealCompleted

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealContract
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000

structure Holds (p : Parameters) (xs : RealMath.Inputs) (pid : Nat) (s₀ : BlockState)
    (i : Nat) (s : BlockState) : Prop where
  execution : RealLoop.Holds p xs pid s₀ i s
  completed : RealCompleted.Holds p xs pid (i/count p) s

/-- Full original-source positive-K real execution, input-only GEMM outputs
for every tile assigned to this PID, and the complete raw-cell memory frame.
The premises constrain original input memory and ordinary shape/stride facts;
there is no prepared-state, partial-sum or assumed-execution premise. -/
theorem executes (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hK : 0<p.K)
    (hBM : 0<p.BM) (hBN : 0<p.BN) (hBK : 0<p.BK)
    (hGM : 0<p.GM) (hs : 0<p.SMS) (hp : s.pids 0<p.SMS)
    (hA : p.C≠p.A) (hB : p.C≠p.B) (hinj : RealOutputAddresses.Injective p)
    (hinput : RealInput.Represents p xs s) :
    ∃ t, stepStmts (RealSource.kernel p Bool.false).body s = some t ∧
      RealCompleted.Holds p xs (s.pids 0) (work p (s.pids 0)) t ∧
      RealInput.Represents p xs t ∧
      ∀ r o, RealLoop.Outside p (s.pids 0) r o → t.mem r o = s.mem r o := by
  have hk : 0<count p := RealMath.blocks_positive p.K p.BK hK hBK
  have hinit : Holds p xs (s.pids 0) s 0 (RealPrefix.result p s) := by
    refine ⟨RealLoop.initial p xs s hinput,?_⟩
    intro q hq
    simp at hq
  have hstep : ∀ i t, i<stop p (s.pids 0) → Holds p xs (s.pids 0) s i t →
      ∃ u, stepStmts (RealProgram.body p) (t.setReg "_i" .nat [] (Tile.scalar i)) = some u ∧
        Holds p xs (s.pids 0) s (i+1) u := by
    intro i t hi h
    let entry := t.setReg "_i" .nat [] (Tile.scalar i)
    have hr := RealLoopState.set_index p xs (s.pids 0) i t h.execution.registers
    have hentry := RealInput.preserved p xs t entry h.execution.input (fun _ => rfl) (fun _ => rfl)
    have he := RealIteration.executes p entry xs (s.pids 0) i hM hN hk hGM hs hp hi
      hentry hr
    obtain ⟨u,hu,hnext⟩ := RealLoop.iteration p xs (s.pids 0) s hM hN hk hGM hs hp hA hB i t hi h.execution
    rw [he] at hu
    have heq := Option.some.inj hu
    subst u
    refine ⟨_,he,hnext,?_⟩
    exact RealCompleted.preserves p entry xs (s.pids 0) i hk hBK hBM hBN hGM hs hp hi hinj h.completed
  obtain ⟨t,he,hfinal⟩ := RealLoopProgram.executes_from_invariant p s
    (Holds p xs (s.pids 0) s) hinit (fun _ _ h => h.execution.bound) hstep
  have hn : stop p (s.pids 0)/count p=work p (s.pids 0) :=
    Nat.mul_div_cancel_left _ hk
  refine ⟨t,he,?_,hfinal.execution.input,hfinal.execution.frame⟩
  simpa only [hn] using hfinal.completed

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealContract
