import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

abbrev RegisterState (α : Type) := FP.Structural.State α

/-- Only source zero literals and identity FP32 loads are bound here. -/
def algebra (p : BmmOptimized.Scalar α) (rest : Algebra α) : Algebra α :=
  { rest with
    literal := fun precision dtype x => if x = 0 then p.zero else rest.literal precision dtype x
    fp32Load := id }

/-- Ascending scalar fused multiply-add sequence, with the live old carry. -/
def fmaFold (p : BmmOptimized.Scalar α) (a b : Nat → α) : Nat → α → α
  | 0,seed => seed
  | n+1,seed => p.fma (a n) (b n) (fmaFold p a b n seed)
def dotTile (p : BmmOptimized.Scalar α) {m k n : Nat}
    (a : TileIndex [m,k] → α) (b : TileIndex [k,n] → α)
    (old : TileIndex [m,n] → α) (i : TileIndex [m,n]) : α :=
  fmaFold p
    (fun j => if h : j<k then a (i.1,⟨j,h⟩,PUnit.unit) else p.zero)
    (fun j => if h : j<k then b (⟨j,h⟩,i.2.1,PUnit.unit) else p.zero)
    k (old i)

def accumulate [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α) {m k n : Nat}
    (left : Op .real [m,k]) (right : Op .real [k,n]) (old : Op .real [m,n])
    (s : RegisterState α) : Option (Values α .real [m,n]) := do
  let a ← evalOp (algebra p rest) (some .fp32) left s
  let b ← evalOp (algebra p rest) (some .fp32) right s
  let seed ← evalOp (algebra p rest) (some .fp32) old s
  return dotTile p a b seed

/-- The observed compiler fusion is localized to the actual accumulator += dot
node. All operands, including the previous accumulator, are evaluated from that
node. Every other statement uses shared Structural semantics. No theorem here
equates unfused FP addition with fused accumulation. -/
def localStep [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α)
    (st : ComputeStmt) (s : RegisterState α) : Option (RegisterState α) :=
  match st with
  | .assign _ _ name (.compute (.alg .fp32 (.add .real broadcast old rhs))) =>
      match rhs with
      | .dot (batch := batch) left right options =>
          match batch with
          | [] =>
              match broadcast with
              | .consSame (.consSame .nil) =>
                  match old with
                  | .ref .real _ oldName =>
                      if name = "o" ∧ oldName = "o" ∧ options = {allowTF32 := some Bool.false} then do
                        let value ← accumulate p rest left right old s
                        return s.setReg name .real _ value
                      else FP.Structural.step (algebra p rest) st s
                  | _ => FP.Structural.step (algebra p rest) st s
              | _ => FP.Structural.step (algebra p rest) st s
          | _ => FP.Structural.step (algebra p rest) st s
      | _ => FP.Structural.step (algebra p rest) st s
  | _ => FP.Structural.step (algebra p rest) st s

theorem localStep_dot [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α)
    {m k n : Nat} (left : Op .real [m,k]) (right : Op .real [k,n]) (s : RegisterState α) :
    localStep p rest
      (.assign .real [m,n] "o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "o")
          (.dot (batch := []) left right {allowTF32 := some Bool.false}))))) s =
      (accumulate p rest left right (.ref .real [m,n] "o") s).bind
        (fun value => some (s.setReg "o" .real [m,n] value)) := by rfl

mutual
def step [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α) :
    ComputeStmt → RegisterState α → Option (RegisterState α)
  | .forRange idx start stop stride body, s => range p rest idx start stop stride body s
  | .forRangeDyn idx first last stride body, s => do
      let a ← evalOp (algebra p rest) none first s
      let b ← evalOp (algebra p rest) none last s
      let c ← evalOp (algebra p rest) none stride s
      range p rest idx (a PUnit.unit) (b PUnit.unit) (c PUnit.unit) body s
  | .ifThenElse cond yes no, s => do
      let choice ← evalOp (algebra p rest) none cond s
      if choice PUnit.unit then run p rest yes s else run p rest no s
  | st, s => localStep p rest st s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

def run [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α) :
    List ComputeStmt → RegisterState α → Option (RegisterState α)
  | [], s => some s
  | st :: tail, s => (step p rest st s).bind (run p rest tail)
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

def range [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α)
    (idx : RegName) (start stop stride : Nat) (body : List ComputeStmt)
    (s : RegisterState α) : Option (RegisterState α) :=
  if stride = 0 then some s else
    if start < stop then do
      let t ← run p rest body (s.setReg idx .nat [] (fun _ => start))
      range p rest idx (start+stride) stop stride body t
    else some s
termination_by (sizeOf body + 1, stop - start)
decreasing_by all_goals (simp_wf; omega)
end

def exec [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α)
    (parameters : Source.Parameters) (s : RegisterState α) : Option (RegisterState α) :=
  run p rest (Source.kernel parameters).surfaceBody s

theorem run_append [Inhabited α] (p : BmmOptimized.Scalar α) (rest : Algebra α)
    (xs ys : List ComputeStmt) (s : RegisterState α) :
    run p rest (xs ++ ys) s = (run p rest xs s).bind (run p rest ys) := by
  induction xs generalizing s with
  | nil => simp [run]
  | cons st xs ih =>
      simp only [List.cons_append, run]
      cases h : step p rest st s <;> simp [h, ih]

#axiomsClean localStep_dot
#axiomsClean run_append
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
