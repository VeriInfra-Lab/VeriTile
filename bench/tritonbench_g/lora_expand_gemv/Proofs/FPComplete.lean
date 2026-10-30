import bench.tritonbench_g.lora_expand_gemv.Proofs.FPExecution

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
open KcacheCopyTriton.TypedCopy (masked_scatter_regs masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem outputOffset_injective (p : Parameters) (initial : State Word) (hcn : 0 < p.outputStride) :
    Function.Injective (outputOffset p initial) := by
  intro i j h
  exact Nat.eq_of_mul_eq_mul_right hcn (Nat.add_left_cancel h)

theorem writeBlock_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial current : State Word) (n : Nat)
    (r : RegionName) (o : Nat) (h : Outside p initial r o) :
    (writeBlock cfg plans profile out p initial current n).mem r o = current.mem r o := by
  unfold writeBlock
  apply State.masked_scatter_frame
  rcases h with hr | ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho ⟨n+i.1.val,hi⟩

theorem writeBlock_readback (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial current : State Word) (n : Nat)
    (hcn : 0 < p.outputStride)
    (hcurrent : ∀ m : Fin p.length, current.mem p.output (outputOffset p initial m.val) =
      if m.val < n then outputCell cfg plans profile out p initial m.val
      else initial.mem p.output (outputOffset p initial m.val)) (m : Fin p.length) :
    (writeBlock cfg plans profile out p initial current n).mem p.output (outputOffset p initial m.val) =
      if m.val < n+p.blockRows then outputCell cfg plans profile out p initial m.val
      else initial.mem p.output (outputOffset p initial m.val) := by
  by_cases hblock : n ≤ m.val ∧ m.val < n+p.blockRows
  · let i : TileIndex [p.blockRows] := (⟨m.val-n, by omega⟩,PUnit.unit)
    have he : n+i.1.val=m.val := by dsimp [i]; omega
    rw [if_pos hblock.2, ← he]
    unfold writeBlock
    apply masked_readback_active
    · rw [he]; exact m.isLt
    · intro a b _ _ haddr
      have hval := outputOffset_injective p initial hcn haddr
      exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)
  · unfold writeBlock
    rw [State.masked_scatter_frame]
    · rw [hcurrent]
      by_cases hn : m.val < n
      · rw [if_pos hn,if_pos (by omega)]
      · have hn' : ¬ m.val < n+p.blockRows := by omega
        rw [if_neg hn,if_neg hn']
    · right
      intro i _ hi he
      have hm := outputOffset_injective p initial hcn he
      have hib := i.1.isLt
      omega

theorem bodyResult_memory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial current : State Word) (n : Nat) :
    (bodyResult cfg plans profile out p initial current n).mem =
      (writeBlock cfg plans profile out p initial current n).mem := by
  simp only [bodyResult,writeBlock,masked_scatter_memory,State.setReg_mem]

theorem bodyInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (hcn : 0 < p.outputStride) (hsep : p.output ≠ p.weight)
    (h : Invariant cfg plans profile out p initial n current) :
    Invariant cfg plans profile out p initial (n+p.blockRows)
      (bodyExecution rest cfg plans profile out p initial current n h).val := by
  change Invariant cfg plans profile out p initial (n+p.blockRows)
    (bodyResult cfg plans profile out p initial current n)
  constructor
  · simp [bodyResult,writeBlock,masked_scatter_regs,h.offk]
  · simp [bodyResult,writeBlock,masked_scatter_regs,h.offn]
  · simp [bodyResult,writeBlock,masked_scatter_regs,h.av]
  · simp [bodyResult,writeBlock,masked_scatter_regs,h.bp]
  · simp [bodyResult,writeBlock,masked_scatter_regs,h.cp]
  · funext o
    rw [bodyResult_memory,writeBlock_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (Ne.symm hsep))]
    exact congrFun h.weights o
  · intro m
    rw [bodyResult_memory]
    exact writeBlock_readback cfg plans profile out p initial current n hcn h.readback m
  · intro r o ho
    rw [bodyResult_memory,writeBlock_frame _ _ _ _ _ _ _ _ _ _ ho]
    exact h.frame r o ho

/-- The entire dynamic output range. No one-tile assumption or fixed trip
count is used; the final partial tile retains its original mask. -/
theorem loopExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (hBN : 0 < p.blockRows) (hcn : 0 < p.outputStride) (hsep : p.output ≠ p.weight)
    (h : Invariant cfg plans profile out p initial n current) :
    ∃ final stop, FP.Structural.range (backend rest cfg plans) "n" n p.length p.blockRows
        (body profile out p) current = some final ∧ p.length ≤ stop ∧
      Invariant cfg plans profile out p initial stop final := by
  have aux : ∀ fuel n (current : State Word), p.length-n=fuel →
      Invariant cfg plans profile out p initial n current →
      ∃ final stop, FP.Structural.range (backend rest cfg plans) "n" n p.length p.blockRows
        (body profile out p) current = some final ∧ p.length ≤ stop ∧
        Invariant cfg plans profile out p initial stop final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro n current he h
      by_cases hn : n < p.length
      · let next := bodyExecution rest cfg plans profile out p initial current n h
        have hi := bodyInvariant rest cfg plans profile out p initial current n hcn hsep h
        obtain ⟨final,stop,hr,hs,hf⟩ := ih (p.length-(n+p.blockRows)) (by omega)
          (n+p.blockRows) next.val rfl hi
        refine ⟨final,stop,?_,hs,hf⟩
        rw [FP.Structural.range,if_neg (by omega),if_pos hn,next.property]
        exact hr
      · refine ⟨current,n,?_,by omega,h⟩
        rw [FP.Structural.range,if_neg (by omega),if_neg hn]
  exact aux _ n current rfl h

def Post (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial final : State Word) : Prop :=
  (∀ m : Fin p.length, final.mem p.output (outputOffset p initial m.val) =
    outputCell cfg plans profile out p initial m.val) ∧
  (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o)

theorem correctness (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial : State Word)
    (hBN : 0 < p.blockRows) (hcn : 0 < p.outputStride) (hsep : p.output ≠ p.weight) :
    ∃ final, FP.Structural.exec (backend rest cfg plans) (source profile out p) initial = some final ∧
      Post cfg plans profile out p initial final := by
  let start := preludeExecution rest cfg plans profile out p initial
  obtain ⟨final,stop,hr,hs,hf⟩ := loopExecution rest cfg plans profile out p initial start.val 0
    hBN hcn hsep (preludeInvariant rest cfg plans profile out p initial)
  refine ⟨final,?_,?_,hf.frame⟩
  · change run (backend rest cfg plans) (source profile out p).surfaceBody initial = some final
    rw [source_decomposition,run_append,start.property]
    simp only [Option.bind_some,run,step,hr]
    rfl
  · intro m
    rw [hf.readback]
    exact if_pos (by omega)

#axiomsClean outputOffset_injective
#axiomsClean writeBlock_frame
#axiomsClean writeBlock_readback
#axiomsClean bodyResult_memory
#axiomsClean bodyInvariant
#axiomsClean loopExecution
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
