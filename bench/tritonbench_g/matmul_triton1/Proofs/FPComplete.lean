import bench.tritonbench_g.matmul_triton1.Proofs.FPExecution
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- All iterations of the original raw K range. The loop count is arbitrary;
K equals BLOCK_K times the accepted number of complete blocks. -/
theorem loopExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (hBK : 0<p.blockInner) (hk : block≤p.steps)
    (h : Invariant P p initial block current) :
    ∃ final, FP.Structural.range (backend rest P) "kk" (block*p.blockInner)
      (inner p) p.blockInner (body p) current=some final ∧ Invariant P p initial p.steps final := by
  have aux : ∀ fuel block (current : State α), p.steps-block=fuel → block≤p.steps →
      Invariant P p initial block current →
      ∃ final, FP.Structural.range (backend rest P) "kk" (block*p.blockInner)
        (inner p) p.blockInner (body p) current=some final ∧ Invariant P p initial p.steps final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro block current he hk h
      by_cases hb : block<p.steps
      · let next := bodyExecution rest P p initial current block h
        obtain ⟨final,hr,hf⟩ := ih (p.steps-(block+1)) (by omega) (block+1) next.val rfl
          (by omega) (bodyInvariant rest P p initial current block h)
        refine ⟨final,?_,hf⟩
        have hraw : block * p.blockInner < inner p := by
          simpa [inner,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb hBK
        rw [FP.Structural.range,if_neg (by omega),if_pos hraw,next.property]
        simpa only [Nat.add_mul,Nat.one_mul] using hr
      · have hb : block=p.steps := by omega
        subst block
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by omega),if_neg (by simp [inner,Nat.mul_comm])]
  exact aux _ block current rfl hk h

def written (P : Primitives α) (p : Parameters) (initial current : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => t.write p.output (cOffset p initial i) (.mk .fp32 (accumulation P p initial p.steps i))) current

def epilogueExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (h : Invariant P p initial p.steps current) :
    {t : State α // run (backend rest P) (epilogue p) current=some t} := by
  refine ⟨written P p initial current,?_⟩
  simp [epilogue, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    h.cptrs,h.acc,written]

omit [Inhabited α] in
private theorem scatter_memory (p : Parameters) (l : List (CIndex p))
    (v : CIndex p → α) (initial current : State α) (h : current.mem=initial.mem) :
    (l.foldl (fun t i => t.write p.output (cOffset p initial i) (.mk .fp32 (v i))) current).mem =
      (l.foldl (fun t i => t.write p.output (cOffset p initial i) (.mk .fp32 (v i))) initial).mem := by
  induction l generalizing initial current with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    -- cOffset depends on launch identifiers, not the changing memory.
    have hp : (current.write p.output (cOffset p initial i) (.mk .fp32 (v i))).mem =
        (initial.write p.output (cOffset p initial i) (.mk .fp32 (v i))).mem := by
      simp [State.write,h]
    exact ih _ _ hp

def Post (P : Primitives α) (p : Parameters) (initial final : State α) : Prop :=
  (∀ i, final.mem p.output (cOffset p initial i)=.mk .fp32 (accumulation P p initial p.steps i)) ∧
  (∀ r o, (r≠p.output ∨ ∀ i : CIndex p, o≠cOffset p initial i) → final.mem r o=initial.mem r o)

omit [Inhabited α] in
theorem output_injective (p : Parameters) (initial : State α) (hN : p.blockColumns≤p.columns) :
    Function.Injective (cOffset p initial) := by
  have heq : cOffset p initial = fun i : CIndex p =>
      (initial.pids 0/columnBlocks p*p.blockRows*p.columns+
        initial.pids 0%columnBlocks p*p.blockColumns)+i.1.val*p.columns+i.2.1.val := by
    funext i
    simp only [cOffset,row,column]
    ring
  rw [heq]
  exact rowMajor2D_inj _ p.columns hN

theorem correctness (rest : Algebra α) (P : Primitives α) (p : Parameters) (initial : State α)
    (hBK : 0<p.blockInner) (hN : p.blockColumns≤p.columns) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧ Post P p initial final := by
  let start := preludeExecution rest P p initial
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P p initial start.val 0 hBK (by omega)
    (preludeInvariant rest P p initial)
  let finish := epilogueExecution rest P p initial middle hi
  have mem : finish.val.mem=(written P p initial initial).mem :=
    scatter_memory p _ _ initial middle hi.memory
  refine ⟨finish.val,?_,?_,?_⟩
  · change run (backend rest P) (source p).surfaceBody initial=some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    have hs (s : State α) : run (backend rest P) [.forRange "kk" 0 (inner p) p.blockInner (body p)] s =
        FP.Structural.range (backend rest P) "kk" 0 (inner p) p.blockInner (body p) s := by simp [run,step]
    simp only [Option.bind_some,hs,Nat.zero_mul] at hm ⊢
    rw [hm]
    exact finish.property
  · intro i
    rw [mem]
    exact State.scatter_readback p.output (cOffset p initial) _ initial i (output_injective p initial hN)
  · intro r o hf
    rw [mem]
    exact State.scatter_frame p.output (cOffset p initial) _ _ r o
      (hf.imp id (fun h i _ => h i)) initial

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean output_injective
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
