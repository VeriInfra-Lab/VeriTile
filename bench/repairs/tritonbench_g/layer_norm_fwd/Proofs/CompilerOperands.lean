import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeParameters

/-! Local source/compiler operand interface. Compiler trees are the expanded
scalar dependencies of emitted registers (moves/shuffles may be expanded by
the ordinary compiler interface). Their constructors name the actual selected
FP32 primitives. Binding is recursive at individual leaves and instructions;
it contains no memory, successful-execution, or numerical output equation. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Compiler
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section

inductive Input where
  | x (second : Bool) (col : Nat)
  | residual (col : Nat)
  | rowscale (second : Bool)
  | weight (second : Bool) (col : Nat)
  | bias (second : Bool) (col : Nat)
  | eps | dropoutP
  deriving DecidableEq

/-- Tags are source identities, not values. A variance leaf's centered value
uses that leaf's own mean owner. Reduction paths are ordered left/right paths. -/
inductive Site where
  | load (input : Input)
  | random (second : Bool) (col : Nat)
  | stream (second : Bool) (col : Nat)
  | combined (col : Nat)
  | meanTree (owner : Nat) (path : List Bool)
  | mean (owner : Nat)
  | centered (owner : Nat)
  | varianceTree (owner : Nat) (path : List Bool)
  | variance (owner : Nat)
  | rstd (owner : Nat)
  | normalized (owner : Nat)
  | affine (second : Bool) (col : Nat)
  | storeY (second : Bool) (col : Nat)
  | storeResidual (col : Nat)
  | storeMean | storeRstd
  deriving DecidableEq

inductive Term where
  | input (input : Input)
  | literal (value : ℝ)
  | random (second : Bool) (col : Nat)
  | cast32 (x : Term)
  | add (x y : Term) | sub (x y : Term) | mul (x y : Term) | div (x y : Term)
  | fma (x y z : Term)
  | sqrt (x : Term)
  | selectGT (x y yes no : Term)
  | at (site : Site) (x : Term)

def Input.value (x : Inputs α) (M row : Nat) (eps p : α) : Input → α
  | .x second col => if second then x.x1 row col else x.x row col
  | .residual col => x.residual row col
  | .rowscale second => x.rowscale (if second then M+row else row)
  | .weight second col => if second then x.weight1 col else x.weight col
  | .bias second col => if second then x.bias1 col else x.bias col
  | .eps => eps | .dropoutP => p

def Term.instantiate (x : Inputs α) (M row : Nat) (eps p : α) : Term → Expr α
  | .input v => .input (v.value x M row eps p)
  | .literal v => .literal v
  | .random second col => .random (x.seeds (if second then M+row else row)) col
  | .cast32 a => .cast32 (a.instantiate x M row eps p)
  | .add a b => .add (a.instantiate x M row eps p) (b.instantiate x M row eps p)
  | .sub a b => .sub (a.instantiate x M row eps p) (b.instantiate x M row eps p)
  | .mul a b => .mul (a.instantiate x M row eps p) (b.instantiate x M row eps p)
  | .div a b => .div (a.instantiate x M row eps p) (b.instantiate x M row eps p)
  | .fma a b c => .fma (a.instantiate x M row eps p) (b.instantiate x M row eps p) (c.instantiate x M row eps p)
  | .sqrt a => .sqrt (a.instantiate x M row eps p)
  | .selectGT a b c d => .selectGT (a.instantiate x M row eps p) (b.instantiate x M row eps p)
      (c.instantiate x M row eps p) (d.instantiate x M row eps p)
  | .at _ a => a.instantiate x M row eps p

/-- Operand trees exported by the compiler interface. `origin` records source
provenance; it is not a hardware instruction. `identity32` retains explicit
FP32 representation casts even when the compiler emits no instruction for it.
Philox7 is a local intrinsic, with seed truncation and counter in `Expr.eval`. -/
inductive Code where
  | read (input : Input)
  | constant32 (value : ℝ)
  | philox7 (second : Bool) (col : Nat)
  | identity32 (x : Code)
  | addF32 (x y : Code) | subF32 (x y : Code) | mulF32 (x y : Code) | divFullF32 (x y : Code)
  | fmaRnF32 (x y z : Code)
  | sqrtApproxFtzF32 (x : Code)
  | selectGreaterF32 (x y yes no : Code)
  | origin (site : Site) (x : Code)

def Code.expression (x : Inputs α) (M row : Nat) (eps p : α) : Code → Expr α
  | .read v => .input (v.value x M row eps p)
  | .constant32 v => .literal v
  | .philox7 second col => .random (x.seeds (if second then M+row else row)) col
  | .identity32 a => .cast32 (a.expression x M row eps p)
  | .addF32 a b => .add (a.expression x M row eps p) (b.expression x M row eps p)
  | .subF32 a b => .sub (a.expression x M row eps p) (b.expression x M row eps p)
  | .mulF32 a b => .mul (a.expression x M row eps p) (b.expression x M row eps p)
  | .divFullF32 a b => .div (a.expression x M row eps p) (b.expression x M row eps p)
  | .fmaRnF32 a b c => .fma (a.expression x M row eps p) (b.expression x M row eps p) (c.expression x M row eps p)
  | .sqrtApproxFtzF32 a => .sqrt (a.expression x M row eps p)
  | .selectGreaterF32 a b c d => .selectGT (a.expression x M row eps p) (b.expression x M row eps p)
      (c.expression x M row eps p) (d.expression x M row eps p)
  | .origin _ a => a.expression x M row eps p

/-- Every constructor checks one exact opcode, ordered operand list, literal,
load identity or source-origin tag. Distinct sites are never selected by
comparing numerical arrays. No algebraic equality appears in this relation. -/
inductive Binds : Term → Code → Prop where
  | input (v) : Binds (.input v) (.read v)
  | literal (v) : Binds (.literal v) (.constant32 v)
  | random (second col) : Binds (.random second col) (.philox7 second col)
  | cast32 : Binds a a' → Binds (.cast32 a) (.identity32 a')
  | add : Binds a a' → Binds b b' → Binds (.add a b) (.addF32 a' b')
  | sub : Binds a a' → Binds b b' → Binds (.sub a b) (.subF32 a' b')
  | mul : Binds a a' → Binds b b' → Binds (.mul a b) (.mulF32 a' b')
  | div : Binds a a' → Binds b b' → Binds (.div a b) (.divFullF32 a' b')
  | fma : Binds a a' → Binds b b' → Binds c c' → Binds (.fma a b c) (.fmaRnF32 a' b' c')
  | sqrt : Binds a a' → Binds (.sqrt a) (.sqrtApproxFtzF32 a')
  | selectGT : Binds a a' → Binds b b' → Binds c c' → Binds d d' →
      Binds (.selectGT a b c d) (.selectGreaterF32 a' b' c' d')
  | at (site) : Binds a a' → Binds (.at site a) (.origin site a')

/-- Structural graph agreement is a theorem of local operand bindings, before
any primitive value is evaluated. It is valid for all opaque inputs, including
NaNs, infinities, signed zero and equal/aliased source values. -/
theorem Binds.expression {t : Term} {code : Code} (h : Binds t code)
    (x : Inputs α) (M row : Nat) (eps p : α) :
    code.expression x M row eps p=t.instantiate x M row eps p := by
  induction h <;> simp_all only [Code.expression,Term.instantiate]

#axiomsClean Binds.expression
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Compiler
