import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Native
/-- Scalar circuits retain operation nesting. There is deliberately no
associativity rule, whole-reduction premise, or whole-output premise. -/
inductive Expr where
  | atom : Nat→Expr
  | zero | one
  | add : Expr→Expr→Expr
  | sub : Expr→Expr→Expr
  | mul : Expr→Expr→Expr
  | div : Expr→Expr→Expr
  | fma : Expr→Expr→Expr→Expr
  | unary : Nat→Expr→Expr

structure Primitives (α : Type) where
  zero : α
  one : α
  add : α→α→α
  sub : α→α→α
  mul : α→α→α
  div : α→α→α
  fma : α→α→α→α
  unary : Nat→α→α

def eval (p : Primitives α) (input : Nat→α) : Expr→α
  | .atom i => input i
  | .zero => p.zero
  | .one => p.one
  | .add a b => p.add (eval p input a) (eval p input b)
  | .sub a b => p.sub (eval p input a) (eval p input b)
  | .mul a b => p.mul (eval p input a) (eval p input b)
  | .div a b => p.div (eval p input a) (eval p input b)
  | .fma a b c => p.fma (eval p input a) (eval p input b) (eval p input c)
  | .unary op a => p.unary op (eval p input a)

/-- Each conditional premise is exactly one scalar numeric operation or local
compiler peephole at its reached operands. Context closure transports the result. -/
inductive Rewrite (p : Primitives α) (input : Nat→α) : Expr→Expr→Prop
  | refl (a) : Rewrite p input a a
  | trans {a b c} : Rewrite p input a b→Rewrite p input b c→Rewrite p input a c
  | addCongr {a b c d} : Rewrite p input a c→Rewrite p input b d→Rewrite p input (.add a b) (.add c d)
  | subCongr {a b c d} : Rewrite p input a c→Rewrite p input b d→Rewrite p input (.sub a b) (.sub c d)
  | mulCongr {a b c d} : Rewrite p input a c→Rewrite p input b d→Rewrite p input (.mul a b) (.mul c d)
  | divCongr {a b c d} : Rewrite p input a c→Rewrite p input b d→Rewrite p input (.div a b) (.div c d)
  | fmaCongr {a b c d e f} : Rewrite p input a d→Rewrite p input b e→Rewrite p input c f→
      Rewrite p input (.fma a b c) (.fma d e f)
  | unaryCongr (op) {a b} : Rewrite p input a b→Rewrite p input (.unary op a) (.unary op b)
  | addSwap (a b) (h : p.add (eval p input a) (eval p input b)=p.add (eval p input b) (eval p input a)) :
      Rewrite p input (.add a b) (.add b a)
  | mulSwap (a b) (h : p.mul (eval p input a) (eval p input b)=p.mul (eval p input b) (eval p input a)) :
      Rewrite p input (.mul a b) (.mul b a)
  | addZeroLeft (a) (h : p.add p.zero (eval p input a)=eval p input a) : Rewrite p input (.add .zero a) a
  | addZeroRight (a) (h : p.add (eval p input a) p.zero=eval p input a) : Rewrite p input (.add a .zero) a
  | divOne (a) (h : p.div (eval p input a) p.one=eval p input a) : Rewrite p input (.div a .one) a
  | fuse (a b c) (h : p.fma (eval p input a) (eval p input b) (eval p input c)=
      p.add (p.mul (eval p input a) (eval p input b)) (eval p input c)) :
      Rewrite p input (.fma a b c) (.add (.mul a b) c)
  | unaryLink (left right) (a) (h : p.unary left (eval p input a)=p.unary right (eval p input a)) :
      Rewrite p input (.unary left a) (.unary right a)

theorem Rewrite.sound (p : Primitives α) (input : Nat→α) {a b : Expr}
    (certificate : Rewrite p input a b) : eval p input a=eval p input b := by
  induction certificate <;> simp_all only [eval]

#axiomsClean Rewrite.sound
end VeriTile.Bench.Repairs.LayerNormWelfold.Native
