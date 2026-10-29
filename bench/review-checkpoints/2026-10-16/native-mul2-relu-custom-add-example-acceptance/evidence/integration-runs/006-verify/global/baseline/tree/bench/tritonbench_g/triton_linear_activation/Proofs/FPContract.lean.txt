import bench.tritonbench_g.triton_linear_activation.Proofs.FPTail
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

theorem execution_memory [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph)
    (fmt saved : Format) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg D F tanh32 erf32 divide) (source g fmt saved p) s=some final ∧
      final.mem=(written cfg g tanh32 erf32 divide fmt saved p s (accumulated cfg D F g fmt p s)).mem := by
  obtain ⟨t,ht,htr⟩ := scalars_run (backend rest cfg D F tanh32 erf32 divide) p s
  obtain ⟨u,hu,hur⟩ := seeds_run rest cfg D F tanh32 erf32 divide g fmt saved p s t htr
  obtain ⟨v,hv,last,hvr⟩ := loop_run rest cfg D F tanh32 erf32 divide g fmt saved p s u 0 _ hur
  let acc := accumulated cfg D F g fmt p s
  have hvready : TailReady p s v acc := ⟨hvr.offm,hvr.offn,hvr.acc⟩
  obtain ⟨w,hw,hwr,hwm⟩ := save_run rest cfg D F tanh32 erf32 divide saved p s v acc hvready hvr.mem
  obtain ⟨x,hx,hxr,hxm⟩ := activation_run rest cfg D F tanh32 erf32 divide g p s w acc hwr
  obtain ⟨final,hf,hfm⟩ := store_run rest cfg D F tanh32 erf32 divide fmt p s x
    (saveWritten cfg saved p s acc) _ hxr (hxm.trans hwm)
  refine ⟨final,?_,hfm⟩
  change run (backend rest cfg D F tanh32 erf32 divide) (source g fmt saved p).surfaceBody s=_
  rw [source_split,suffix_split]
  simp only [run_append,ht,hu,Option.bind_some,run,step,hv]
  simpa only [Option.bind,bind,pure,hw,hx] using hf

def Post [Inhabited Word] (cfg : FP.Config) (g : Graph) (tanh32 erf32 : Word → Word)
    (divide : Word → Word → Word) (fmt saved : Format) (p : Parameters)
    (s final : State Word) (a : Accumulator p) : Prop :=
  (∀ i, active p s i=Bool.true → final.mem p.output (outputOffset p s i)=
    outputCell fmt (castWord cfg .single fmt (activate cfg g tanh32 erf32 divide p.activation (a i)))) ∧
  (p.saveInputs=Bool.true → p.saved≠p.output → Function.Injective (savedOffset p s) →
    ∀ i, final.mem p.saved (savedOffset p s i)=outputCell saved (castWord cfg .single saved (a i))) ∧
  (∀ r o, (r≠p.output ∨ ∀ i, active p s i=Bool.true → o≠outputOffset p s i) →
    (p.saveInputs=Bool.false ∨ r≠p.saved ∨ ∀ i, o≠savedOffset p s i) → final.mem r o=s.mem r o)

theorem written_post [Inhabited Word] (cfg : FP.Config) (g : Graph) (tanh32 erf32 : Word → Word)
    (divide : Word → Word → Word) (fmt saved : Format) (p : Parameters) (s : State Word)
    (a : Accumulator p) (hinj : Function.Injective (outputOffset p s)) :
    Post cfg g tanh32 erf32 divide fmt saved p s (written cfg g tanh32 erf32 divide fmt saved p s a) a := by
  refine ⟨?_,?_,?_⟩
  · intro i hi
    exact State.masked_scatter_readback _ _ _ (fun i => active p s i=Bool.true) _ i hi hinj
  · intro hs hne hsave i
    unfold written storeResult
    rw [State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ _ _ (Or.inl hne)]
    simp only [saveWritten,hs,if_true]
    exact State.scatter_readback _ _ _ _ i hsave
  · intro r o hc hs
    unfold written storeResult
    rw [State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ _ _
      (hc.imp_right fun hh i _ => hh i)]
    rcases hs with hs | hs
    · simp [saveWritten,hs]
    · unfold saveWritten
      split
      · exact State.scatter_frame _ _ _ _ r o (hs.imp_right fun hh i _ => hh i) s
      · rfl

theorem correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph)
    (fmt saved : Format) (p : Parameters) (s : State Word)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest cfg D F tanh32 erf32 divide) (source g fmt saved p) s=some final ∧
      Post cfg g tanh32 erf32 divide fmt saved p s final (accumulated cfg D F g fmt p s) := by
  obtain ⟨final,he,hm⟩ := execution_memory rest cfg D F tanh32 erf32 divide g fmt saved p s
  refine ⟨final,he,?_⟩
  have hp := written_post cfg g tanh32 erf32 divide fmt saved p s (accumulated cfg D F g fmt p s) hinj
  simpa only [Post,hm] using hp
#axiomsClean execution_memory
#axiomsClean written_post
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
