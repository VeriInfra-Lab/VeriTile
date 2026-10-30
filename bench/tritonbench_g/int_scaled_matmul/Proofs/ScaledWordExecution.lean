import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordReference
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance scaledExecutionDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local simp 2000] FP.Structural.eval_intToSigned
private theorem twoIndex_eta {m n : Nat} (i : TileIndex [m,n]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨r,c,⟨⟩⟩
  rfl
private theorem pointer_step {m n : Nat} (r : RegionName) (f : TileIndex [m,n] → Nat) (a b : Nat) :
    bop (fun a b => (a.1,a.2+b)) Broadcast.scalarR (fun i => (r,f i))
      (bop (fun x y => x*y) Broadcast.nil (fun _ => a) (fun _ => b)) =
        (fun i => (r,f i+a*b)) := by
  funext i
  rcases i with ⟨x,y,⟨⟩⟩
  rfl
private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]
structure Invariant (p : Parameters) (initial : State Word) (k : Nat) (current : State Word) : Prop where
  memory : current.mem = initial.mem
  undefined : current.undef = initial.undef
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "rk" = some (fun i => inputInner p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "A" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "B" =
    some (fun i => (p.weight, bOffset p initial k i))
  acc : current.regs .int [p.blockRows,p.blockColumns] "acc" = some (accumulation p initial k)

def preludeExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest cfg u) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, original, typedScaled, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend]
  rfl

theorem preludeInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) :
    Invariant p s 0 (preludeExecution rest cfg u p s).val := by
  constructor <;>
    simp [preludeExecution, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow,
      aOffset, bOffset, min_lt_form, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, show (0.0 : ℝ)=0 by norm_num]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]


def bodyResult (p : Parameters) (initial current : State Word) (k : Nat) : State Word :=
  let s1 := (current.setReg "j" .nat [] (fun _ => k)).setReg "k" .nat [] (fun _ => p.inner-k*p.blockInner)
  let s2 := s1.setReg "a" .int [p.blockRows,p.blockInner] (aValue p initial k)
  let s3 := s2.setReg "b" .int [p.blockInner,p.blockColumns] (bValue p initial k)
  let s4 := s3.setReg "acc" .int [p.blockRows,p.blockColumns] (accumulation p initial (k+1))
  let s5 := s4.setReg "A" .ptr [p.blockRows,p.blockInner]
    (fun i => (p.input,aOffset p initial k i+p.blockInner*p.inputInnerStride))
  s5.setReg "B" .ptr [p.blockInner,p.blockColumns]
    (fun i => (p.weight,bOffset p initial k i+p.blockInner*p.weightInnerStride))

def bodyExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant p initial k current) :
    {t : State Word // run (backend rest cfg u) (body p)
      (current.setReg "j" .nat [] (fun _ => k)) = some t} := by
  refine ⟨bodyResult p initial current k, ?_⟩
  cases he : p.even <;>
    simp [body, originalBody, original, typedScaled, widths, widthAssignment, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
      integralFloorDiv, integralMod, h.offsk, h.aptrs, h.bptrs, h.acc, h.memory,
      backend, IntegerExpr.sameShape, he, bodyResult, accumulation]
  all_goals rw [show aValue p initial k = (fun i => Int.bmod
      (if p.even=Bool.true ∨ inputInner p initial i.2.1 < p.inner-k*p.blockInner then
        (initial.mem p.input (aOffset p initial k i)).read .int else 0) 256) from by
    funext j
    simp [aValue,wrap]]
  all_goals rw [show bValue p initial k = (fun i => Int.bmod
      (if p.even=Bool.true ∨ inputInner p initial i.1 < p.inner-k*p.blockInner then
        (initial.mem p.weight (bOffset p initial k i)).read .int else 0) 256) from by
    funext j
    simp [bValue,wrap]]
  all_goals simp only [he, Bool.false_eq_true, false_or, true_or, ↓reduceIte]
  all_goals simp only [wrap, BitVec.toInt_ofInt, twoIndex_eta, pointer_step]
  all_goals rfl

theorem bodyInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant p initial k current) :
    Invariant p initial (k+1) (bodyExecution rest cfg u p initial current k h).val := by
  constructor <;>
    simp [bodyExecution, bodyResult, h.memory, h.undefined, h.pidm, h.pidn, h.offsk,
      aOffset, bOffset, Nat.add_mul, Nat.add_assoc]


#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
