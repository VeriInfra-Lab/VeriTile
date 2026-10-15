import VeriTile.Triton.DSL.Expansion.Common
import VeriTile.Triton.Core.Integer

/-! Expansion helpers for direct, shape-preserving Philox tuple calls.

The source-profile pass owns machine-width admission and explicit helper alias
bindings. These functions receive admitted operands, preserve their independent
shapes, and capture every input before assigning any tuple result. The reserved
capture registers are implementation locals; source identifiers using their
prefix are rejected by that pass. Their deterministic names are shared by
source/candidate expansions and never depend on a macro expansion's fresh scope.
-/
open Lean

namespace VeriTile.Triton.DSL

def directRandomCapturePrefix : String := "__veritile_direct_random_"

/-- Build the actual sequence of source broadcasts. Zero rounds preserve four
distinct shapes; later rounds broadcast only the two XOR chains of that round. -/
def directRandomPlan : Nat → SInfo → SInfo → SInfo → SInfo →
    MacroM (TSyntax `term × Array SInfo)
  | 0, c0, c1, c2, c3 =>
      return (← `(Random.Direct.Plan.done), #[c0, c1, c2, c3])
  | n + 1, c0, c1, c2, c3 => do
      let (bc21, a) ← broadcastTerm c2 c1 "tl.philox counter 2/1"
      let (bc03, b) ← broadcastTerm c0 c3 "tl.philox counter 0/3"
      let (rest, outputs) ← directRandomPlan n a c2 b c0
      return (← `(Random.Direct.Plan.step $bc21 $bc03 $rest), outputs)

/-- All operands have the signed mathematical carrier expected by the core;
the separate profile retains each original signed/unsigned machine dtype.
The emitted block first evaluates all five operands into private registers,
then computes the four fields from those captured values. In particular,
`c0, c1, c2, c3 = tl.philox(seed, c0, c1, c2, c3)` is simultaneous. -/
def expandDirectPhilox (env : Env) (targets : Array (TSyntax `ident))
    (cfg : TSyntax `term) (rounds : Nat) (inputs : Array EOut) :
    MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
  unless targets.size == 4 && inputs.size == 5 do
    Macro.throwError "tl.philox requires four tuple targets and five operands"
  ensureShape SInfo.scalar inputs[0]!.shape "tl.philox seed must be scalar"
  for input in inputs do
    ensureDType .int input.dtype "tl.philox operand"
    ensureAlgorithmOnly "tl.philox operand" input
  let (plan, outputs) ← directRandomPlan rounds
    inputs[1]!.shape inputs[2]!.shape inputs[3]!.shape inputs[4]!.shape
  let input0 ← inputs[1]!.shape.term
  let input1 ← inputs[2]!.shape.term
  let input2 ← inputs[3]!.shape.term
  let input3 ← inputs[4]!.shape.term
  let output0 ← outputs[0]!.term
  let output1 ← outputs[1]!.term
  let output2 ← outputs[2]!.term
  let output3 ← outputs[3]!.term
  let inputShapes ← `(Random.Direct.Shapes.mk $input0 $input1 $input2 $input3)
  let outputShapes ← `(Random.Direct.Shapes.mk $output0 $output1 $output2 $output3)
  let roundCount : TSyntax `term := ⟨Syntax.mkNumLit (toString rounds)⟩
  let mut algorithms : Array (TSyntax `term) := #[]
  let mut computes : Array (TSyntax `term) := #[]
  let mut captured : Array (TSyntax `term) := #[]
  for index in [:5] do
    let input := inputs[index]!
    let shape ← input.shape.term
    let name := directRandomCapturePrefix ++ toString index
    unless !(env.any (fun entry => entry.1 == name)) &&
        !(targets.any (fun target => target.getId.toString == name)) do
      Macro.throwError "tl.philox capture register collides with a source identifier"
    let register : TSyntax `term := ⟨Syntax.mkStrLit name⟩
    algorithms := algorithms.push (← `(Stmt.assign .int $shape $register $input.term))
    computes := computes.push
      (← `(ComputeStmt.assign .int $shape $register (ComputeExpr.alg $input.term)))
    captured := captured.push (← `(Op.ref .int $shape $register))
  let fields : Array (TSyntax `term) := #[
    ← `(Random.Direct.Field.c0), ← `(Random.Direct.Field.c1),
    ← `(Random.Direct.Field.c2), ← `(Random.Direct.Field.c3)]
  let seed := captured[0]!
  let c0 := captured[1]!
  let c1 := captured[2]!
  let c2 := captured[3]!
  let c3 := captured[4]!
  let mut outputEnv := env
  for index in [:4] do
    let target := targets[index]!
    let name ← identAsStr target
    let shape ← outputs[index]!.term
    let field := fields[index]!
    let value ← `(Op.philox4x32 (ins := $inputShapes) (outs := $outputShapes)
      (rounds := $roundCount) $cfg $plan $field $seed $c0 $c1 $c2 $c3)
    algorithms := algorithms.push (← `(Stmt.assign .nat $shape $name $value))
    computes := computes.push
      (← `(ComputeStmt.assign .nat $shape $name (ComputeExpr.alg $value)))
    outputEnv := (target.getId.toString, .nat, outputs[index]!, none) :: outputEnv
  return (← `(Stmt.ifThen (Op.constBool Bool.true) [$algorithms,*]),
    ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$computes,*]), outputEnv, Bool.false)

/-- The canonical helper includes the original integer conversion rounding
and multiplication rounding. It is not replaced by exact real arithmetic. -/
def expandDirectUniform32 (input : EOut) : MacroM EOut := do
  ensureDType .nat input.dtype "uint_to_uniform_float requires an admitted uint32 input"
  ensureAlgorithmOnly "uint_to_uniform_float input" input
  let value ← `(Op.uniform32 $input.term)
  return ⟨value, .real, input.shape, some (← fp32ComputeExpr value), some .fp32⟩

/-- Native unsigned arithmetic retains the low bits of the signed mathematical
sum/product. This also handles a negative signed operand without clamping it. -/
def expandDirectUnsignedArithmetic (width : Nat) (multiply : Bool)
    (left right : EOut) : MacroM EOut := do
  ensureDType .int left.dtype "unsigned arithmetic lhs"
  ensureDType .int right.dtype "unsigned arithmetic rhs"
  ensureAlgorithmOnly "unsigned arithmetic lhs" left
  ensureAlgorithmOnly "unsigned arithmetic rhs" right
  let (bc, shape) ← broadcastTerm left.shape right.shape "unsigned source arithmetic"
  let arithmetic ← if multiply then
    `(Op.mul NumericDType.int $bc $left.term $right.term)
  else
    `(Op.add NumericDType.int $bc $left.term $right.term)
  let bits : TSyntax `term := ⟨Syntax.mkNumLit (toString width)⟩
  let value ← `(Op.intToUnsigned $bits $arithmetic)
  return ⟨value, .nat, shape, none, none⟩

end VeriTile.Triton.DSL
