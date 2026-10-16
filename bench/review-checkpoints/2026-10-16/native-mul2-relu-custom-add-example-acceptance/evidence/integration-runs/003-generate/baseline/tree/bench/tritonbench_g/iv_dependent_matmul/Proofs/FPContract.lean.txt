import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPExecution
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

theorem execution_memory [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg D F) (source g p) s=some final ∧
      final.mem=(written cfg p s (accumulated cfg D F g p s)).mem := by
  obtain ⟨t,ht,htr⟩ := prologue_run rest cfg D F p s
  obtain ⟨u,hu,hur⟩ := loop_run rest cfg D F g p s t 0 _ htr
  obtain ⟨final,hf,hfm⟩ := suffix_run rest cfg D F p s u _ hur
  refine ⟨final,?_,hfm⟩
  change run (backend rest cfg D F) (source g p).surfaceBody s=_
  rw [source_split]
  simp only [run_append,ht,Option.bind_some,run,step,stopOp,evalOp_unfold,
    numeric,bop,Option.bind,bind,pure]
  change ((do let u ← FP.Structural.range (backend rest cfg D F) "k" 0 (count p) 1 (body g p) t
              pure u).bind (run (backend rest cfg D F) (suffix p)))=some final
  simp only [hu,Option.bind,bind,pure]
  exact hf

def Post [Inhabited Word] (cfg : FP.Config) (p : Parameters) (s final : State Word)
    (a : Accumulator p) : Prop :=
  (∀ i, active p s i=Bool.true → final.mem p.output (outputOffset p s i)=.mk .fp32 (outputValues cfg p a i)) ∧
  (∀ r o, (r≠p.output ∨ ∀ i, active p s i=Bool.true → o≠outputOffset p s i) → final.mem r o=s.mem r o)

theorem written_post [Inhabited Word] (cfg : FP.Config) (p : Parameters) (s : State Word)
    (a : Accumulator p) (hinj : Function.Injective (outputOffset p s)) :
    Post cfg p s (written cfg p s a) a := by
  constructor
  · intro i hi
    exact State.masked_scatter_readback _ _ _ (fun i => active p s i=Bool.true) _ i hi hinj
  · intro r o ho
    exact State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ r o
      (ho.imp_right fun hh i _ => hh i) s

theorem correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s : State Word) (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest cfg D F) (source g p) s=some final ∧
      Post cfg p s final (accumulated cfg D F g p s) := by
  obtain ⟨final,he,hm⟩ := execution_memory rest cfg D F g p s
  refine ⟨final,he,?_⟩
  have hp := written_post cfg p s (accumulated cfg D F g p s) hinj
  simpa only [Post,hm] using hp

theorem output_injective (p : Parameters) (s : State Word)
    (hStride : p.outputColumnStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    Function.Injective (outputOffset p s) := by
  have heq : outputOffset p s = fun i : TileIndex [p.blockRows,p.blockColumns] =>
      (p.outputRowStride*(rowBlock p s*p.blockRows)+colBlock p s*p.blockColumns)+
        i.1.val*p.outputRowStride+i.2.1.val := by
    funext i
    simp only [outputOffset,row,col,hStride,Nat.one_mul]
    ring
  rw [heq]
  exact rowMajor2D_inj _ _ hWidth
#axiomsClean execution_memory
#axiomsClean written_post
#axiomsClean correctness
#axiomsClean output_injective
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
