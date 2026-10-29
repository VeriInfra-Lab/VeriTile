import bench.tritonbench_g.matmul_triton2.Proofs.FPExecution
namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loopExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (hk : block≤p.steps)
    (h : Invariant P p initial block current) :
    ∃ final, FP.Structural.range (backend rest P) "k" block p.steps 1 (body p) current=some final ∧
      Invariant P p initial p.steps final := by
  have aux : ∀ fuel block (current : State α), p.steps-block=fuel → block≤p.steps →
      Invariant P p initial block current →
      ∃ final, FP.Structural.range (backend rest P) "k" block p.steps 1 (body p) current=some final ∧
        Invariant P p initial p.steps final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro block current he hk h
      by_cases hb : block<p.steps
      · let next := bodyExecution rest P p initial current block h
        obtain ⟨final,hr,hf⟩ := ih (p.steps-(block+1)) (by omega) (block+1) next.val rfl
          (by omega) (bodyInvariant rest P p initial current block h)
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by decide),if_pos hb,next.property]
        exact hr
      · have hb : block=p.steps := by omega
        subst block
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by decide),if_neg (by omega)]
  exact aux _ block current rfl hk h

theorem loop_count (p : Parameters) (hBK : 0<p.blockInner) :
    (inner p+p.blockInner-1)/p.blockInner=p.steps := by
  unfold inner
  rw [show p.blockInner*p.steps+p.blockInner-1=(p.blockInner-1)+p.steps*p.blockInner by rw [Nat.mul_comm p.blockInner p.steps]; omega,
    Nat.add_mul_div_right _ _ hBK,Nat.div_eq_of_lt (by omega),Nat.zero_add]

theorem dynamicLoop (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (current : State α) (hBK : 0<p.blockInner) :
    step (backend rest P) (loopStmt p) current =
      FP.Structural.range (backend rest P) "k" 0 p.steps 1 (body p) current := by
  simp [loopStmt,source,typed,ComputeKernel.surfaceBody,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,bop,integralFloorDiv,Option.bind,ComputeDType.eraseDType,
    body,loop_count p hBK]

def epilogueExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (h : Invariant P p initial p.steps current) :
    {t : State α // run (backend rest P) (epilogue p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [epilogue,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.offm,h.offn,h.acc]
  rfl

def written (P : Primitives α) (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then t.write p.output (cOffset p initial i)
      (.mk .fp32 (accumulation P p initial p.steps i)) else t) initial

theorem epilogueMemory (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (h : Invariant P p initial p.steps current) :
    (epilogueExecution rest P p initial current h).val.mem=(written P p initial).mem := by
  simp [epilogueExecution,written,masked_scatter_memory,h.memory,active,cOffset,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def Post (P : Primitives α) (p : Parameters) (initial final : State α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (cOffset p initial i)=
    .mk .fp32 (accumulation P p initial p.steps i)) ∧
  (∀ r o, (r≠p.output ∨ ∀ i : CIndex p, active p initial i → o≠cOffset p initial i) →
    final.mem r o=initial.mem r o)

theorem correctness (rest : Algebra α) (P : Primitives α) (p : Parameters) (initial : State α)
    (hBK : 0<p.blockInner) (hInj : Function.Injective (cOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧ Post P p initial final := by
  let start := preludeExecution rest P p initial
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P p initial start.val 0 (by omega)
    (preludeInvariant rest P p initial)
  let finish := epilogueExecution rest P p initial middle hi
  have mem := epilogueMemory rest P p initial middle hi
  refine ⟨finish.val,?_,?_,?_⟩
  · change run (backend rest P) (source p).surfaceBody initial=some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    have hs (s : State α) : run (backend rest P) [loopStmt p] s =
        FP.Structural.range (backend rest P) "k" 0 p.steps 1 (body p) s := by
      simp [run,dynamicLoop rest P p s hBK]
    simp only [Option.bind_some,hs,hm]
    exact finish.property
  · intro i ha
    rw [mem]
    exact masked_readback_active _ _ _ _ _ i ha (fun a b _ _ he => hInj he)
  · intro r o hf
    rw [mem]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hf.imp id (fun h i _ => h i)) _

/-- The row-major conditions of the old IO headline imply the original
arbitrary-stride summary's address injectivity condition. -/
theorem rowMajor_injective (p : Parameters) (initial : State α)
    (hcn : p.outputColumn=1) (hBN : p.blockColumns≤p.outputRow) :
    Function.Injective (cOffset p initial) := by
  have heq : cOffset p initial = fun i : CIndex p =>
      (rowBlock p initial*p.blockRows*p.outputRow+columnBlock p initial*p.blockColumns)+
        i.1.val*p.outputRow+i.2.1.val := by
    funext i
    simp only [cOffset,row,column,hcn]
    ring
  rw [heq]
  exact rowMajor2D_inj _ p.outputRow hBN

#axiomsClean loopExecution
#axiomsClean loop_count
#axiomsClean dynamicLoop
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness
#axiomsClean rowMajor_injective
end
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
