import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.MatmulTriton1.MaskHoist
open VeriTile.Triton FP.Structural Revision001.Core
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prelude (p : Parameters) := (Kernel.maskHoisted p).surfaceBody.take 12
def loopStmt (p : Parameters) := ((Kernel.maskHoisted p).surfaceBody[12]?).getD (.ifThen (.constBool Bool.false) [])
def body (p : Parameters) : List ComputeStmt := match (Kernel.maskHoisted p).surfaceBody[12]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (Kernel.maskHoisted p).surfaceBody.drop 13

theorem source_decomposition (p : Parameters) : (Kernel.maskHoisted p).surfaceBody =
    prelude p ++ [loopStmt p] ++ epilogue p := rfl

structure Invariant (P : Primitives α) (p : Parameters) (initial : State α)
    (block : Nat) (current : State α) : Prop extends Revision001.Core.Invariant P p initial block current where
  rowMask : current.regs .bool [p.blockRows] "row_mask" = some (fun i => decide (row p initial i.1<p.rows))
  columnMask : current.regs .bool [p.blockColumns] "column_mask" = some (fun i => decide (column p initial i.1<p.columns))

private theorem min_lt_form (a b : Nat) : min a b = if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

def preludeExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest P) (prelude p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,Kernel.maskHoisted,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,backend]
  rfl

theorem preludeInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    Invariant P p s 0 (preludeExecution rest P p s).val := by
  constructor
  · constructor <;> simp [preludeExecution,backend,accumulation,aOffset,bOffset,
    row,column,rowBlock,columnBlock,blocksM,blocksN,min_lt_form,numericLt,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,show (0.0 : ℝ)=0 by norm_num]
    all_goals funext i
    all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · simp [preludeExecution,row,column,rowBlock,columnBlock,blocksN,numericLt,natLt,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    rfl
  · simp [preludeExecution,row,column,rowBlock,columnBlock,blocksN,numericLt,natLt,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    rfl

def bodyResult (P : Primitives α) (p : Parameters) (initial current : State α) (block : Nat) : State α :=
  let s1 := current.setReg "k" .nat [] (fun _ => block)
  let s2 := s1.setReg "a" .real [p.blockRows,p.blockInner] (aValues P p initial block)
  let s3 := s2.setReg "b" .real [p.blockInner,p.blockColumns] (bValues P p initial block)
  let s4 := s3.setReg "accumulator" .real [p.blockRows,p.blockColumns] (accumulation P p initial (block+1))
  let s5 := s4.setReg "a_ptrs" .ptr [p.blockRows,p.blockInner]
    (fun i => (p.input,aOffset p initial block i+p.blockInner*p.inputInner))
  s5.setReg "b_ptrs" .ptr [p.blockInner,p.blockColumns]
    (fun i => (p.weight,bOffset p initial block i+p.blockInner*p.weightInner))

def bodyExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (h : Invariant P p initial block current) :
    {t : State α // run (backend rest P) (body p)
      (current.setReg "k" .nat [] (fun _ => block))=some t} := by
  refine ⟨bodyResult P p initial current block,?_⟩
  simp [body,Kernel.maskHoisted,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,h.aptrs,h.bptrs,h.offm,h.offn,h.offk,h.acc,h.memory,h.rowMask,h.columnMask,backend,
    bodyResult,accumulation,aValues,bValues,Broadcast.leftIndex,Broadcast.rightIndex,
    show (0.0 : ℝ)=0 by norm_num]
  rfl

theorem bodyInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (h : Invariant P p initial block current) :
    Invariant P p initial (block+1) (bodyExecution rest P p initial current block h).val := by
  constructor
  · constructor <;> simp [bodyExecution,bodyResult,h.memory,h.offm,h.offn,h.offk,h.rowMask,h.columnMask,
    aOffset,bOffset,Nat.add_mul,Nat.add_assoc]

  · simp [bodyExecution,bodyResult,h.rowMask]
  · simp [bodyExecution,bodyResult,h.columnMask]

theorem loopExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (hk : block≤steps p)
    (h : Invariant P p initial block current) :
    ∃ final, FP.Structural.range (backend rest P) "k" block (steps p) 1 (body p) current=some final ∧
      Invariant P p initial (steps p) final := by
  have aux : ∀ fuel block (current : State α), steps p-block=fuel → block≤steps p →
      Invariant P p initial block current →
      ∃ final, FP.Structural.range (backend rest P) "k" block (steps p) 1 (body p) current=some final ∧
        Invariant P p initial (steps p) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro block current he hk h
      by_cases hb : block<steps p
      · let next := bodyExecution rest P p initial current block h
        obtain ⟨final,hr,hf⟩ := ih (steps p-(block+1)) (by omega) (block+1) next.val rfl
          (by omega) (bodyInvariant rest P p initial current block h)
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by decide),if_pos hb,next.property]
        exact hr
      · have hb : block=steps p := by omega
        subst block
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by decide),if_neg (by omega)]
  exact aux _ block current rfl hk h

theorem loop_count (p : Parameters) (hBK : 0<p.blockInner) :
    (inner p+p.blockInner-1)/p.blockInner=steps p := by rfl

theorem dynamicLoop (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (current : State α) (hBK : 0<p.blockInner) :
    step (backend rest P) (loopStmt p) current =
      FP.Structural.range (backend rest P) "k" 0 (steps p) 1 (body p) current := by
  simp [loopStmt,Kernel.maskHoisted,ComputeKernel.surfaceBody,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,bop,integralFloorDiv,Option.bind,ComputeDType.eraseDType,
    body,loop_count p hBK]
  rfl

def epilogueExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (h : Invariant P p initial (steps p) current) :
    {t : State α // run (backend rest P) (epilogue p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [epilogue,Kernel.maskHoisted,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.offm,h.offn,h.acc,h.rowMask,h.columnMask]
  rfl

theorem epilogueMemory (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (h : Invariant P p initial (steps p) current) :
    (epilogueExecution rest P p initial current h).val.mem=(Revision001.Core.written P p initial).mem := by
  simp [epilogueExecution,Revision001.Core.written,masked_scatter_memory,h.memory,active,cOffset,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]


theorem exec_memory (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (positive : 0<p.blockInner) :
    ∃ final, exec (backend rest P) (Kernel.maskHoisted p) initial=some final ∧
      final.mem=(Revision001.Core.written P p initial).mem := by
  let start := preludeExecution rest P p initial
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P p initial start.val 0 (by omega)
    (preludeInvariant rest P p initial)
  let finish := epilogueExecution rest P p initial middle hi
  refine ⟨finish.val,?_,epilogueMemory rest P p initial middle hi⟩
  change run _ (Kernel.maskHoisted p).surfaceBody initial=some finish.val
  rw [source_decomposition,run_append,run_append,start.property]
  have hs (s : State α) : run (backend rest P) [loopStmt p] s =
      FP.Structural.range (backend rest P) "k" 0 (steps p) 1 (body p) s := by
    simp [run,dynamicLoop rest P p s positive]
  simp only [Option.bind_some,hs,hm]
  exact finish.property

#axiomsClean exec_memory
end
end VeriTile.Bench.Repairs.MatmulTriton1.MaskHoist
