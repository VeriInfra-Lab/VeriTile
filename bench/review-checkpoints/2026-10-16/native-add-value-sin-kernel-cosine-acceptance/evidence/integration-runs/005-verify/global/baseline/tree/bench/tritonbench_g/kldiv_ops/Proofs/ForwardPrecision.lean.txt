import bench.tritonbench_g.kldiv_ops.Proofs.BackwardPrecision

namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def forwardBody (source : ComputeKernel) : List ComputeStmt :=
  match (source.surfaceBody[6]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []

namespace ForwardPrecision

/-- Independent expected load: storage precision, signed offset and explicit
masked zero are all retained. These templates never inspect a source AST. -/
def expectedLoad (storage : Storage) (name pointer : RegName) (B : Nat) : ComputeStmt :=
  let address : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] pointer)
    (.castIntToNat (.ref .int [B] "offsets"))
  let mask : Op .bool [B] := .ref .bool [B] "mask"
  match storage with
  | .fp32 => .assign .real [B] name (.compute (.load .fp32 (.ptr address)
      (.maskOther mask ((Op.const 0.0).broadcast [B]))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp64 (.load .real (.ptr address)
      (.maskOther mask ((Op.const 0.0).broadcast [B])))))
  | .fp16 => .assign .fp16 [B] name (.alg (.load .fp16 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))
  | .bf16 => .assign .bf16 [B] name (.alg (.load .bf16 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))
  | .f8e4 => .assign .f8e4 [B] name (.alg (.load .f8e4 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B]))))
  | .f8e5 => .assign .f8e5 [B] name (.alg (.load .f8e5 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B]))))

def isWide (pred target : Storage) : Bool := pred == .fp64 || target == .fp64

/-- A loaded operand keeps its source-local context before the surrounding
fp32/fp64 arithmetic promotion. -/
def promotedOperand (storage : Storage) (wide : Bool) (name : RegName)
    (B : Nat) : Op .real [B] :=
  match storage with
  | .fp64 => .ref .real [B] name
  | .fp32 => if wide then .withPrecision .real .fp32 (.ref .real [B] name)
      else .ref .real [B] name
  | .fp16 => .castFloat .fp16 .real (if wide then
      .withPrecision .fp16 .fp32 (.ref .fp16 [B] name) else .ref .fp16 [B] name)
  | .bf16 => .castFloat .bf16 .real (if wide then
      .withPrecision .bf16 .fp32 (.ref .bf16 [B] name) else .ref .bf16 [B] name)
  | .f8e4 => .castFloat .f8e4 .real (if wide then
      .withPrecision .f8e4 .fp32 (.ref .f8e4 [B] name) else .ref .f8e4 [B] name)
  | .f8e5 => .castFloat .f8e5 .real (if wide then
      .withPrecision .f8e5 .fp32 (.ref .f8e5 [B] name) else .ref .f8e5 [B] name)

/-- Runtime Python-float epsilon enters at fp32. With an fp64 target, its fp32
cast remains scoped inside the fp64 clamp; narrow targets are promoted to fp32. -/
def expectedClamp (target : Storage) (B : Nat) (eps : ℝ) : Op .real [B] :=
  let value := promotedOperand target Bool.false "y_true" B
  let epsilon : Op .real [] := .castFloat .real .real (.const eps)
  let epsilon := if target == .fp64 then .withPrecision .real .fp32 epsilon else epsilon
  .maximumFloat .real .scalarR value epsilon

def expectedDefaultFormula (pred target : Storage) (B : Nat) (eps : ℝ) : ComputeStmt :=
  let wide := isWide pred target
  let y := promotedOperand pred wide "y" B
  let t := promotedOperand target wide "y_true" B
  let logarithm := Op.log (expectedClamp target B eps)
  let logarithm := if target != .fp64 && wide then
      .withPrecision .real .fp32 logarithm else logarithm
  let expression := Op.mul .real (.consSame .nil) t
    (.sub .real (.consSame .nil) logarithm y)
  .assign .real [B] "loss" (if wide then .compute (.alg .fp64 expression)
    else .compute (.alg .fp32 expression))

/-- The exponential stays at the target's precision before multiplication with
the independently promoted subtraction. -/
def expectedLogFormula (pred : Storage) (target : MathStorage) (B : Nat) : ComputeStmt :=
  let wide := isWide pred target.storage
  let y := promotedOperand pred wide "y" B
  let t := promotedOperand target.storage wide "y_true" B
  let exponential := Op.exp (.ref .real [B] "y_true")
  let exponential := if target == .fp32 && wide then
      .withPrecision .real .fp32 exponential else exponential
  let expression := Op.mul .real (.consSame .nil) exponential
    (.sub .real (.consSame .nil) t y)
  .assign .real [B] "loss" (if wide then .compute (.alg .fp64 expression)
    else .compute (.alg .fp32 expression))

def outputMask (dtype : TileDType) (shape : TileShape) (mask : Option (Op .bool shape)) :
    MaskOpt dtype shape := match mask with
  | none => .none
  | some m => .mask m

def expectedStore (wide : Bool) (output : Storage) (shape : TileShape)
    (address : Op .ptr shape) (mask : Option (Op .bool shape)) : ComputeStmt :=
  let raw : Op .real shape := .ref .real shape "loss"
  let value : Op .real shape := if wide then .withPrecision .real .fp64 raw
    else .withPrecision .real .fp32 raw
  match output with
  | .fp32 => .store .real shape (.ptr address)
      (.compute (.alg .fp32 (if wide then .castFloat .real .real value else raw)))
      (outputMask .real shape mask)
  | .fp64 => .store .real shape (.ptr address)
      (.compute (.alg .fp64 (if wide then raw else .castFloat .real .real
        (.withPrecision .real .fp32 raw)))) (outputMask .real shape mask)
  | .fp16 => .store .fp16 shape (.ptr address) (.alg (.castFloat .real .fp16 value))
      (outputMask .fp16 shape mask)
  | .bf16 => .store .bf16 shape (.ptr address) (.alg (.castFloat .real .bf16 value))
      (outputMask .bf16 shape mask)
  | .f8e4 => .store .f8e4 shape (.ptr address) (.alg (.castFloat .real .f8e4 value))
      (outputMask .f8e4 shape mask)
  | .f8e5 => .store .f8e5 shape (.ptr address) (.alg (.castFloat .real .f8e5 value))
      (outputMask .f8e5 shape mask)

/-- The reduction and addition use the loss precision. The fp32 initializer is
preserved even in the formal fp64-carry profiles excluded by SourceDomain. -/
def expectedAccumulate (wide : Bool) (B : Nat) : ComputeStmt :=
  let raw : Op .real [] := .ref .real [] "loss_sum"
  let accumulator := if wide then .withPrecision .real .fp32 raw else raw
  let sum : Op .real [] := .reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false
    (.ref .real [B] "loss")
  let expression := Op.add .real .nil accumulator sum
  .assign .real [] "loss_sum" (if wide then .compute (.alg .fp64 expression)
    else .compute (.alg .fp32 expression))

def expectedBranch (wide : Bool) (output : Storage) (B reduction : Nat) : ComputeStmt :=
  .ifThenElse (.eq .nat .nil (.constNat reduction) (.constNat 0))
    [expectedStore wide output [B]
      (.ptrAdd .scalarL (.ref .ptr [] "O") (.castIntToNat (.ref .int [B] "offsets")))
      (some (.ref .bool [B] "mask"))]
    [expectedAccumulate wide B]

/-- The source stores the fp32 carried accumulator once after the whole loop,
with the declared destination conversion and no elementwise mask. -/
def expectedFinalStore (output : Storage) : ComputeStmt :=
  let raw : Op .real [] := .ref .real [] "loss_sum"
  match output with
  | .fp32 => .store .real [] (.ptr (.ref .ptr [] "O")) (.alg raw) .none
  | .fp64 => .store .real [] (.ptr (.ref .ptr [] "O"))
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 raw)))) .none
  | .fp16 => .store .fp16 [] (.ptr (.ref .ptr [] "O"))
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 raw))) .none
  | .bf16 => .store .bf16 [] (.ptr (.ref .ptr [] "O"))
      (.alg (.castFloat .real .bf16 (.withPrecision .real .fp32 raw))) .none
  | .f8e4 => .store .f8e4 [] (.ptr (.ref .ptr [] "O"))
      (.alg (.castFloat .real .f8e4 (.withPrecision .real .fp32 raw))) .none
  | .f8e5 => .store .f8e5 [] (.ptr (.ref .ptr [] "O"))
      (.alg (.castFloat .real .f8e5 (.withPrecision .real .fp32 raw))) .none

def expectedFinalBranch (output : Storage) (reduction : Nat) : ComputeStmt :=
  .ifThen (.ne .nat .nil (.constNat reduction) (.constNat 0))
    [expectedFinalStore output]

/-- Exact independent load, clamp/log/subtract/multiply, store, reduction/carry,
and post-loop conversion templates for all 216 formal profiles and both variants.
Frontend-admitted reductions additionally require ForwardDefaultSupported. -/
theorem default_profiles (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    let source := forwardDefault pred target output v Y T O sy st so N B eps reduction
    let body := forwardBody source
    body[2]? = some (expectedLoad pred "y" "Y" B) ∧
    body[3]? = some (expectedLoad target "y_true" "T" B) ∧
    body[4]? = some (expectedDefaultFormula pred target B eps) ∧
    body[5]? = some (expectedBranch (isWide pred target) output B reduction) ∧
    source.surfaceBody[7]? = some (expectedFinalBranch output reduction) := by
  cases pred <;> cases target <;> cases output <;> cases v <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Exact independent exp/subtract/multiply and all store/reduction templates
for all 72 formal log-target profiles, including both implementation variants. -/
theorem log_profiles (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    let source := forwardLog pred target output v Y T O sy st so N B eps reduction
    let body := forwardBody source
    body[2]? = some (expectedLoad pred "y" "Y" B) ∧
    body[3]? = some (expectedLoad target.storage "y_true" "T" B) ∧
    body[4]? = some (expectedLogFormula pred target B) ∧
    body[5]? = some (expectedBranch (isWide pred target.storage) output B reduction) ∧
    source.surfaceBody[7]? = some (expectedFinalBranch output reduction) := by
  cases pred <;> cases target <;> cases output <;> cases v <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

end ForwardPrecision

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
