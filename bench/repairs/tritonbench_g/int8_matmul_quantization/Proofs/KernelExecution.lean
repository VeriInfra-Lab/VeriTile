import bench.repairs.tritonbench_g.int8_matmul_quantization.Kernels
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Core
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000

/-- Integer register representation. The native instance uses signed32 words;
the mathematical instance uses mathematical integers. This is a representation
law, not a numerical-experiment or whole-result premise. -/
structure IntegerCarrier (I : Type) where
  encode : Int → I
  decode : I → Int
  roundtrip : ∀ x, encode (decode x) = x

def signed32 : IntegerCarrier (BitVec 32) :=
  ⟨BitVec.ofInt 32, BitVec.toInt, fun _ => BitVec.ofInt_toInt⟩

def mathematicalIntegers : IntegerCarrier Int := ⟨id, id, fun _ => rfl⟩

/-- A fixed structural instruction plan at each source coordinate. The
selection receives only dimensions/coordinates, never payloads or memory. -/
abbrev Plans := (width : Nat) → Nat → Nat → Nat → Nat → DotPlan width

/-- Evaluate exactly the source dot operands and previous accumulator. The
fixed K32/scalar plan preserves the selected fused or separate instruction
sequence, and no callback receives a full row, kernel result or output state. -/
def accumulate {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) {m k n : Nat}
    (left : Op .int [m,k]) (right : Op .int [k,n]) (old : Op .int [m,n])
    (s : State F) : Option (Values F .int [m,n]) := do
  let xs ← evalOp M none left s
  let ys ← evalOp M none right s
  let previous ← evalOp M none old s
  let pm ← s.regs .nat [] "pid_m"
  let pn ← s.regs .nat [] "pid_n"
  let iteration ← s.regs .nat [] "k"
  return fun i => carrier.decode <|
    (plans k (pm PUnit.unit*m+i.1.val) (pn PUnit.unit*n+i.2.1.val)
      (s.pids 1) (iteration PUnit.unit)).eval ops
      (fun j => BitVec.ofInt 8 (xs (i.1,j,PUnit.unit)))
      (fun j => BitVec.ofInt 8 (ys (j,i.2.1,PUnit.unit)))
      (carrier.encode (previous i))

/-- The only arithmetic extension recognizes the actual accumulator += dot
assignment. Other expressions keep shared Structural semantics. In particular
we do not pretend a zero-seed dot can model the observed carry-fed saturating
MMA. Both operands and the old accumulator are evaluated from this expression.
The normal register/type/missing-value failure behavior is retained. -/
def localStep {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (st : ComputeStmt) (s : State F) : Option (State F) :=
  match st with
  | .assign .int _ name (.alg (.add .int broadcast old rhs)) =>
      match rhs with
      | .dotInt (batch := batch) left right options =>
          match batch with
          | [] =>
              match broadcast with
              | .consSame (.consSame .nil) =>
                  match old with
                  | .ref .int _ oldName =>
                      if name = "accumulator" ∧ oldName = "accumulator" ∧ options = {} then do
                        let v ← accumulate M carrier ops plans left right old s
                        return s.setReg name .int _ v
                      else FP.Structural.step M st s
                  | _ => FP.Structural.step M st s
              | _ => FP.Structural.step M st s
          | _ => FP.Structural.step M st s
      | _ => FP.Structural.step M st s
  | _ => FP.Structural.step M st s

theorem localStep_dot {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) {m k n : Nat} (left : Op .int [m,k]) (right : Op .int [k,n])
    (s : State F) :
    localStep M carrier ops plans
      (.assign .int [m,n] "accumulator" (.alg (.add .int (.consSame (.consSame .nil))
        (.ref .int [m,n] "accumulator") (.dotInt (batch := []) left right)))) s =
      (accumulate M carrier ops plans left right (.ref .int [m,n] "accumulator") s).bind
        (fun v => some (s.setReg "accumulator" .int [m,n] v)) := by rfl

theorem localStep_load {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) {sh : TileShape} (name : RegName)
    (mem : MemAccess .int sh) (mask : MaskOpt .int sh) (s : State F) :
    localStep M carrier ops plans (.assign .int sh name (.alg (.load .int mem mask))) s =
      FP.Structural.step M (.assign .int sh name (.alg (.load .int mem mask))) s := by rfl

theorem localStep_pointer {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) {sh : TileShape} (name : RegName) (e : ComputeExpr .ptr sh) (s : State F) :
    localStep M carrier ops plans (.assign .ptr sh name e) s =
      FP.Structural.step M (.assign .ptr sh name e) s := by rfl

mutual
def step {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) : ComputeStmt → State F → Option (State F)
  | .forRange idx start stop stride body, s => range M carrier ops plans idx start stop stride body s
  | .ifThenElse cond yes no, s => do
      let choice ← evalOp M none cond s
      if choice PUnit.unit then run M carrier ops plans yes s else run M carrier ops plans no s
  | st, s => localStep M carrier ops plans st s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

def run {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) : List ComputeStmt → State F → Option (State F)
  | [], s => some s
  | st :: rest, s => (step M carrier ops plans st s).bind (run M carrier ops plans rest)
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

def range {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (idx : RegName) (start stop stride : Nat) (body : List ComputeStmt)
    (s : State F) : Option (State F) :=
  if stride = 0 then some s else
    if start < stop then do
      let t ← run M carrier ops plans body (s.setReg idx .nat [] (fun _ => start))
      range M carrier ops plans idx (start+stride) stop stride body t
    else some s
termination_by (sizeOf body + 1, stop - start)
decreasing_by all_goals (simp_wf; omega)
end

def exec {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (kernel : ComputeKernel) (s : State F) : Option (State F) :=
  run M carrier ops plans kernel.surfaceBody s

theorem run_append {I F H : Type} [Inhabited F]
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (xs ys : List ComputeStmt) (s : State F) :
    run M carrier ops plans (xs ++ ys) s =
      (run M carrier ops plans xs s).bind (run M carrier ops plans ys) := by
  induction xs generalizing s with
  | nil => simp [run]
  | cons st xs ih =>
      simp only [List.cons_append, run]
      cases h : step M carrier ops plans st s <;> simp [h, ih]

#axiomsClean signed32
#axiomsClean mathematicalIntegers
#axiomsClean localStep_dot
#axiomsClean run_append
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelExecution
