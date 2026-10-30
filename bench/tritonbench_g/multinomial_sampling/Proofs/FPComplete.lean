import bench.tritonbench_g.multinomial_sampling.Proofs.FPPrologue
import bench.tritonbench_g.multinomial_sampling.Proofs.FPBody
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem before_count (p : Parameters) (k : Nat) : k<count p ↔ k*p.blockTokens<p.tokens := by
  have hn := p.tokensDomain.1
  change k+1≤(p.tokens+p.blockTokens-1)/p.blockTokens ↔ _
  rw [Nat.le_div_iff_mul_le hn]
  simp only [Nat.add_mul,Nat.one_mul]
  omega

/-- Entire original raw range, including a partially masked final token block. -/
theorem loopExecution (rest : Algebra α) (P : Primitives α) (d : Storage) (p : Parameters)
    (initial current : State α) (k : Nat) (hk : k≤count p)
    (h : Invariant P p initial (rounds P d p initial k) current) :
    ∃ final, FP.Structural.range (backend rest P) "b_idx" (k*p.blockTokens) p.tokens
      p.blockTokens (body d p) current=some final ∧
      Invariant P p initial (rounds P d p initial (count p)) final := by
  have aux : ∀ fuel k (current : State α), count p-k=fuel → k≤count p →
      Invariant P p initial (rounds P d p initial k) current →
      ∃ final, FP.Structural.range (backend rest P) "b_idx" (k*p.blockTokens) p.tokens
        p.blockTokens (body d p) current=some final ∧
        Invariant P p initial (rounds P d p initial (count p)) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro k current he hk h
      by_cases hb : k<count p
      · let next := bodyResult P d p initial current (k*p.blockTokens) (rounds P d p initial k)
        have hi : Invariant P p initial (rounds P d p initial (k+1)) next :=
          bodyInvariant P d p initial current (k*p.blockTokens) _ h
        obtain ⟨final,hr,hf⟩ := ih (count p-(k+1)) (by omega) (k+1) next rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by have := p.tokensDomain.1; omega),
          if_pos ((before_count p k).mp hb),bodyExecution rest P d p initial current (k*p.blockTokens) _ h]
        simpa only [Nat.add_mul,Nat.one_mul] using hr
      · have hb : k=count p := by omega
        subst k
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by have := p.tokensDomain.1; omega),
          if_neg (by simpa only [←before_count] using Nat.lt_irrefl (count p))]
  exact aux _ k current rfl hk h

def writeBack (P : Primitives α) (d : Storage) (p : Parameters) (initial current : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun t i =>
    if active p initial i then t.write p.output (row p initial i)
      (.mk .int ((rounds P d p initial (count p)).output i)) else t) current

def epilogueExecution (rest : Algebra α) (P : Primitives α) (d : Storage) (p : Parameters)
    (initial current : State α)
    (h : Invariant P p initial (rounds P d p initial (count p)) current) :
    {t : State α // run (backend rest P) (epilogue d p) current=some t} := by
  refine ⟨writeBack P d p initial current,?_⟩
  cases d <;> simp [epilogue,source,original,original16,originalBfloat16,original64,
    ComputeKernel.surfaceBody,Random.i32ArangeSize,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,store,
    h.off,h.mask,h.output,writeBack,Region.cast,Option.bind]

private theorem masked_mem_eq {ι : Type} (r : RegionName) (off : ι → Nat) (v : ι → Cell α)
    (enabled : ι → Prop) [DecidablePred enabled] (l : List ι) (initial current : State α)
    (h : current.mem=initial.mem) :
    (l.foldl (fun t i => if enabled i then t.write r (off i) (v i) else t) current).mem =
      (l.foldl (fun t i => if enabled i then t.write r (off i) (v i) else t) initial).mem := by
  induction l generalizing current initial with
  | nil => exact h
  | cons i l ih =>
    simp only [List.foldl_cons]
    by_cases hi : enabled i
    · simp only [hi,↓reduceIte]
      apply ih
      simp [State.write,h]
    · simpa only [hi,↓reduceIte] using ih initial current h

theorem output_injective (p : Parameters) (initial : State α) : Function.Injective (row p initial) := by
  intro a b he
  apply Prod.ext
  · apply Fin.ext
    simp only [row] at he
    omega
  · exact Subsingleton.elim _ _

theorem correctness (rest : Algebra α) (P : Primitives α) (d : Storage) (p : Parameters)
    (initial : State α) (hseed : SeedDomain p initial) :
    ∃ final, FP.Structural.exec (backend rest P) (source d p) initial=some final ∧
      final.mem=(written P d p initial).mem ∧
      (∀ i : Row p, active p initial i → final.mem p.output (row p initial i)=
        .mk .int ((rounds P d p initial (count p)).output i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : Row p, active p initial i → o≠row p initial i) →
        final.mem r o=initial.mem r o) := by
  let start := preludeExecution rest P d p initial hseed
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P d p initial start.val 0 (by omega)
    (preludeInvariant rest P d p initial hseed)
  let finish := epilogueExecution rest P d p initial middle hi
  have hmem : finish.val.mem=(written P d p initial).mem :=
    masked_mem_eq p.output (row p initial) _ _ _ initial middle hi.memory
  refine ⟨finish.val,?_,hmem,?_,?_⟩
  · change run (backend rest P) (source d p).surfaceBody initial=some finish.val
    rw [source_decomposition]
    have hs (s : State α) :
        run (backend rest P) [.forRange "b_idx" 0 p.tokens p.blockTokens (body d p)] s =
          FP.Structural.range (backend rest P) "b_idx" 0 p.tokens p.blockTokens (body d p) s := by simp [run,step]
    simp only [Nat.zero_mul] at hm
    simp only [run_append,start.property,Option.bind_some,hs,hm,finish.property]
  · intro i ha
    rw [hmem]
    exact State.masked_scatter_readback _ _ _ _ _ _ ha (output_injective p initial)
  · intro r o hf
    rw [hmem]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hf.imp id (fun h i _ => h i)) _

#axiomsClean before_count
#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
