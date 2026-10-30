import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
theorem forward_parts (p : Parameters) :
    (forward p).surfaceBody=(fwdParts p).before ++
      [.forRangeDyn "i" (.constNat 0) (.constNat p.NT) (.constNat 1) (fwdParts p).body] := rfl

theorem fwdLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (s : State α) (ctx : FwdReady p pid input s)
    (rows : FwdRows p 0 s) (hc : FwdCarry p (fun _ => zero M) s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i" 0 p.NT 1 (fwdParts p).body s=some t ∧
      FwdReady p pid input t ∧ FwdRows p p.NT t ∧
      FwdCarry p (fwdCarry M D p pid input p.NT) t ∧
      t.mem=fwdMemory M D p pid input s.mem := by
  let update := fun n mem => writeOutput M p pid n
    (fwdOutput M D p pid input n (fwdCarry M D p pid input n)) mem
  let I := fun n (t : State α) => FwdReady p pid input t ∧ FwdRows p n t ∧
    FwdCarry p (fwdCarry M D p pid input n) t ∧ t.mem=foldRange update 0 n 1 s.mem
  have step : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (fwdParts p).body
        (t.setReg "i" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t hi
    let run := fwdBodyExecution M D p pid input t n (fwdCarry M D p pid input n) hi.1 hi.2.1 hi.2.2.1
    have next := fwdBody_ready M D p pid input t n (fwdCarry M D p pid input n) hi.1 hi.2.1 hi.2.2.1
    refine ⟨run.val,run.property,next.1,next.2.1,next.2.2,?_⟩
    change (fwdBodyExecution M D p pid input t n _ _ _ _).val.mem=_
    rw [fwdBody_memory,hi.2.2.2]
    exact (foldRange_succ update n s.mem).symm
  have initial : I 0 s := ⟨ctx,rows,hc,by simp [foldRange_zero]⟩
  exact range_invariant (engine (withDot M D)) "i" (fwdParts p).body p.NT I step 0 (Nat.zero_le _) s initial

/-- Whole forward loop from zero carry, with current-memory stores and preserved
input regions. No supplied carry, output or successful-execution premise. -/
theorem forwardProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (input : Memory α) (s : State α) (hin : Inputs s.mem input) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (forward p) s=some t ∧
      t.mem=fwdMemory M D p ⟨s.pids 0,s.pids 1⟩ input s.mem ∧ Inputs t.mem input := by
  let start := fwdEntry M D p s
  have he := fwdEntry_ready M D p s
  have ctx : FwdReady p ⟨s.pids 0,s.pids 1⟩ input start.val :=
    {he.1 with inputs:=he.1.inputs.trans hin}
  obtain ⟨finish,hloop,hctx,hrows,hcarry,hmem⟩ := fwdLoop M D p ⟨s.pids 0,s.pids 1⟩ input start.val ctx he.2.1 he.2.2
  refine ⟨finish,?_,?_,hctx.inputs⟩
  · change run (engine (withDot M D)) (forward p).surfaceBody s=some finish
    rw [forward_parts,run_append]
    simpa only [start.property,Option.bind_some,run,step,evalOp_unfold,bind,
      Option.bind,hloop]
  · rw [hmem]
    dsimp only [start]
    rw [fwdEntry_memory]
#axiomsClean forwardProgram
end
end VeriTile.Bench.Repairs.LightningAttention
