import bench.repairs.tritonbench_g.chunk_linear_attn.OutputExecution
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem output_parts (p : Parameters) (scale : ℝ) :
    (forwardO p scale).surfaceBody=(outputParts p scale).before ++
      [.forRangeDyn "i_k" (outputParts p scale).start (outputParts p scale).stop
        (outputParts p scale).stride (outputParts p scale).body] ++ (outputParts p scale).after := rfl

theorem outputLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p (outputInitial M p) s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i_k" 0 (keyBlocks p) 1 (outputParts p scale).body s=some t ∧
      OutputReady p pid input t ∧ HasOutput p (outputStates M D p pid input (keyBlocks p)) t := by
  let I := fun n (t : State α) => OutputReady p pid input t ∧ HasOutput p (outputStates M D p pid input n) t
  have hs : ∀ n, n<keyBlocks p → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (outputParts p scale).body (t.setReg "i_k" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t hi
    let u := outputBody M D p scale pid input t n (outputStates M D p pid input n) hi.1 hi.2
    exact ⟨u.val,u.property,outputBody_ready M D p scale pid input t n _ hi.1 hi.2⟩
  exact range_invariant (engine (withDot M D)) "i_k" (outputParts p scale).body (keyBlocks p)
    I hs 0 (Nat.zero_le _) s ⟨ctx,ha⟩

theorem outputProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (forwardO p scale) s=some t ∧
      t.mem=outputMemory M D p ⟨s.pids 0,s.pids 1,s.pids 2⟩ s.mem s.mem scale := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := outputEntry M D p scale s
  have hs := outputEntry_ready M D p scale s
  obtain ⟨mid,hm,hctx,ha⟩ := outputLoop M D p scale pid s.mem start.val hs.1 hs.2
  let finish := outputFinish M D p scale pid s.mem mid (outputStates M D p pid s.mem (keyBlocks p)) hctx ha
  refine ⟨finish.val,?_,?_⟩
  · have h0 : (outputParts p scale).start=.constNat 0 := rfl
    have hn : (outputParts p scale).stop=.div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK) := rfl
    have h1 : (outputParts p scale).stride=.constNat 1 := rfl
    have hl : step (engine (withDot M D))
        (.forRangeDyn "i_k" (outputParts p scale).start (outputParts p scale).stop
          (outputParts p scale).stride (outputParts p scale).body) start.val=some mid := by
      simpa [step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,keyBlocks] using hm
    change run (engine (withDot M D)) (forwardO p scale).surfaceBody s=some finish.val
    rw [output_parts,List.append_assoc,run_append]
    simp only [start.property,Option.bind_some,List.singleton_append,run,hl,bind,finish.property]
  · exact outputFinish_memory M D p scale pid s.mem mid _ hctx ha

#axiomsClean outputLoop
#axiomsClean outputProgram
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
