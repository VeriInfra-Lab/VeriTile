import bench.tritonbench_g.matmul_triton1.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def preludeExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest P) (prelude p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend]
  rfl

theorem preludeInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    Invariant P p s 0 (preludeExecution rest P p s).val := by
  constructor <;> simp [preludeExecution, backend, accumulation, aOffset, bOffset, cOffset,
    row, column, columnBlocks, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    show (0.0 : ℝ)=0 by norm_num]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def bodyResult (P : Primitives α) (p : Parameters) (initial current : State α) (block : Nat) : State α :=
  let s1 := current.setReg "kk" .nat [] (fun _ => block*p.blockInner)
  let s2 := s1.setReg "x_sub" .fp32 [p.blockRows,p.blockInner] (aValues p initial block)
  let s3 := s2.setReg "y_sub" .fp32 [p.blockInner,p.blockColumns] (bValues p initial block)
  let s4 := s3.setReg "z" .fp32 [p.blockRows,p.blockColumns] (accumulation P p initial (block+1))
  let s5 := s4.setReg "x_ptrs" .ptr [p.blockRows,p.blockInner]
    (fun i => (p.input,aOffset p initial block i+p.blockInner))
  s5.setReg "y_ptrs" .ptr [p.blockInner,p.blockColumns]
    (fun i => (p.weight,bOffset p initial block i+p.blockInner*p.columns))

def bodyExecution (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (h : Invariant P p initial block current) :
    {t : State α // run (backend rest P) (body p)
      (current.setReg "kk" .nat [] (fun _ => block*p.blockInner))=some t} := by
  refine ⟨bodyResult P p initial current block,?_⟩
  simp [body, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, h.aptrs, h.bptrs, h.acc, h.memory,
    backend, bodyResult, accumulation, aValues, bValues,
    Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

theorem bodyInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial current : State α) (block : Nat) (h : Invariant P p initial block current) :
    Invariant P p initial (block+1) (bodyExecution rest P p initial current block h).val := by
  constructor <;> simp [bodyExecution, bodyResult, h.memory, h.cptrs,
    aOffset,bOffset,Nat.add_mul,Nat.add_assoc]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
