import bench.tritonbench_g.chunk_linear_attn.Proofs.PairOutputLoop

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem program (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (forwardOHoisted p scale) s=some t ∧
      t.mem=outputMemory M D p ⟨s.pids 0,s.pids 1,s.pids 2⟩ s.mem s.mem scale := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  have hs := entry_ready M D p scale s
  obtain ⟨mid,hm,hctx,ha,hh⟩ := loop M D p scale pid s.mem (entry M D p scale s) hs.1 hs.2.1 hs.2.2
  let finish := outputFinish M D p scale pid s.mem mid (outputStates M D p pid s.mem (keyBlocks p)) hctx ha
  refine ⟨finish.val,?_,outputFinish_memory M D p scale pid s.mem mid _ hctx ha⟩
  have h0 : (parts p scale).start=.constNat 0 := rfl
  have hn : (parts p scale).stop=.div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK) := rfl
  have h1 : (parts p scale).stride=.constNat 1 := rfl
  have hl : step (engine (withDot M D))
      (.forRangeDyn "i_k" (parts p scale).start (parts p scale).stop
        (parts p scale).stride (parts p scale).body) (entry M D p scale s)=some mid := by
    simpa [step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,keyBlocks] using hm
  change run (engine (withDot M D)) (forwardOHoisted p scale).surfaceBody s=some finish.val
  rw [decomposition,List.append_assoc,run_append,entry_run]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,after,finish.property]
#axiomsClean program

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
