import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.FPReference
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
private theorem min_lt_form (a b : Nat) : min a b = if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

def preludeExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest P) (prelude p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,backend]
  rfl

theorem preludeInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    Invariant P p s 0 (preludeExecution rest P p s).val := by
  constructor <;> simp [preludeExecution,backend,accumulation,aOffset,bOffset,
    row,column,rowBlock,columnBlock,blocksM,blocksN,min_lt_form,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,show (0.0 : ℝ)=0 by norm_num]
  all_goals funext i
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

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
  simp [body,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,h.aptrs,h.bptrs,h.offm,h.offn,h.offk,h.acc,h.memory,backend,
    bodyResult,accumulation,aValues,bValues,Broadcast.leftIndex,Broadcast.rightIndex,
    show (0.0 : ℝ)=0 by norm_num]
  rfl

theorem bodyInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (h : Invariant P p initial block current) :
    Invariant P p initial (block+1) (bodyExecution rest P p initial current block h).val := by
  constructor <;> simp [bodyExecution,bodyResult,h.memory,h.offm,h.offn,h.offk,
    aOffset,bOffset,Nat.add_mul,Nat.add_assoc]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
