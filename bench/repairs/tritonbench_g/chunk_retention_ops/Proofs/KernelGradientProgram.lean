import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientFinish
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul load rawLoad dot scatter scatter_frame transpose valueBlocks range_invariant foldRange_succ foldRange_zero)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem gradient_parts (p : Parameters) (scale : ℝ) :
    (kernel p scale).surfaceBody=(gradientParts p scale).before ++
      [.forRangeDyn "i_v" (gradientParts p scale).start (gradientParts p scale).stop
        (gradientParts p scale).stride (gradientParts p scale).body] ++ (gradientParts p scale).after := rfl

theorem gradientLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p (gradInitial M p) s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i_v" 0 (valueBlocks p) 1 (gradientParts p scale).body s=some t ∧
      GradientReady p pid heads input t ∧ GradientFixed M D p pid input scale t ∧
      HasGrad p (gradStates M D p pid input scale (valueBlocks p)) t ∧
      t.mem=foldRange (fun vb mem => gradValueWrite M p pid heads vb (gradValue M D p pid input scale vb) mem)
        0 (valueBlocks p) 1 s.mem := by
  let write := fun vb mem => gradValueWrite M p pid heads vb (gradValue M D p pid input scale vb) mem
  let I := fun n (t : State α) => GradientReady p pid heads input t ∧ GradientFixed M D p pid input scale t ∧
    HasGrad p (gradStates M D p pid input scale n) t ∧ t.mem=foldRange write 0 n 1 s.mem
  have hs : ∀ n, n<valueBlocks p → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (gradientParts p scale).body (t.setReg "i_v" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t hi
    let u := gradientBody M D p scale pid heads input t n (gradStates M D p pid input scale n) hi.1 hi.2.1 hi.2.2.1
    have hu := gradientBody_values M D p scale pid heads input t n _ hi.1 hi.2.1 hi.2.2.1
    refine ⟨u.val,u.property,hu.1,hu.2.1,hu.2.2,?_⟩
    change (gradientBody M D p scale pid heads input t n _ hi.1 hi.2.1 hi.2.2.1).val.mem=_
    rw [gradientBody_memory,hi.2.2.2]
    exact (foldRange_succ write n s.mem).symm
  have h0 : I 0 s := ⟨ctx,fixed,ha,by simp [foldRange_zero]⟩
  exact range_invariant (engine (withDot M D)) "i_v" (gradientParts p scale).body
    (valueBlocks p) I hs 0 (Nat.zero_le _) s h0

theorem gradientProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) (hBT : p.BT≤2^31) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel p scale) s=some t ∧
      t.mem=gradientMemory M D p ⟨s.pids 0,s.pids 1,s.pids 2⟩ (s.numPids 2) s.mem s.mem scale := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := gradientEntry M D p scale s
  have hs := gradientEntry_values M D p scale s hBT
  obtain ⟨mid,hm,hctx,hfixed,ha,hmem⟩ := gradientLoop M D p scale pid (s.numPids 2) s.mem start.val hs.1 hs.2.1 hs.2.2.1
  let finish := gradientFinish M D p scale pid (s.numPids 2) s.mem mid
    (gradStates M D p pid s.mem scale (valueBlocks p)) hctx hfixed ha
  refine ⟨finish.val,?_,?_⟩
  · have h0 : (gradientParts p scale).start=.constNat 0 := rfl
    have hn : (gradientParts p scale).stop=.div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat p.V) (.constNat p.BV)) (.constNat 1)) (.constNat p.BV) := rfl
    have h1 : (gradientParts p scale).stride=.constNat 1 := rfl
    have hl : step (engine (withDot M D))
        (.forRangeDyn "i_v" (gradientParts p scale).start (gradientParts p scale).stop
          (gradientParts p scale).stride (gradientParts p scale).body) start.val=some mid := by
      simpa [step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,valueBlocks] using hm
    change run (engine (withDot M D)) (kernel p scale).surfaceBody s=some finish.val
    rw [gradient_parts,List.append_assoc,run_append]
    simp only [start.property,Option.bind_some,List.singleton_append,run,hl,bind,finish.property]
  · change (gradientFinish M D p scale pid (s.numPids 2) s.mem mid _ hctx hfixed ha).val.mem=_
    rw [gradientFinish_memory,hmem,hs.2.2.2]
    rfl

#axiomsClean gradientLoop
#axiomsClean gradientProgram
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
