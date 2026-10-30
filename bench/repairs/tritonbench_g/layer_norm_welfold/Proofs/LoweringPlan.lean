import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ScalarTransport
import Mathlib.Data.Nat.Pairing
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Native

inductive LocalStep where
  | addSwap : Expr→Expr→LocalStep
  | mulSwap : Expr→Expr→LocalStep
  | addZeroLeft : Expr→LocalStep
  | addZeroRight : Expr→LocalStep
  | divOne : Expr→LocalStep
  | fuse : Expr→Expr→Expr→LocalStep
  | unaryLink : Nat→Nat→Expr→LocalStep
  | castErase : Nat→Expr→LocalStep
  | load : Nat→Nat→LocalStep
  | literal : Expr→LocalStep

/-- All fields concern a single reached scalar instruction. `used` is computed
from the value-erased lowering plan, not supplied as a numeric output oracle. -/
structure LocalLaws (p : Primitives α) (input : Nat→α) (used : LocalStep→Prop) : Prop where
  addSwap : ∀ a b, used (.addSwap a b) → p.add (eval p input a) (eval p input b)=p.add (eval p input b) (eval p input a)
  mulSwap : ∀ a b, used (.mulSwap a b) → p.mul (eval p input a) (eval p input b)=p.mul (eval p input b) (eval p input a)
  addZeroLeft : ∀ a, used (.addZeroLeft a) → p.add p.zero (eval p input a)=eval p input a
  addZeroRight : ∀ a, used (.addZeroRight a) → p.add (eval p input a) p.zero=eval p input a
  divOne : ∀ a, used (.divOne a) → p.div (eval p input a) p.one=eval p input a
  fuse : ∀ a b c, used (.fuse a b c) → p.add (p.mul (eval p input a) (eval p input b)) (eval p input c)=p.fma (eval p input a) (eval p input b) (eval p input c)
  unaryLink : ∀ left right a, used (.unaryLink left right a) → p.unary left (eval p input a)=p.unary right (eval p input a)
  castErase : ∀ op a, used (.castErase op a) → p.unary op (eval p input a)=eval p input a
  load : ∀ kind col, used (.load kind col) → input (Nat.pair kind col)=p.unary 3 (input (Nat.pair kind col))
  literal : ∀ a, used (.literal a) → eval p input a=p.unary 4 (eval p input a)

/-- A syntax-only compiler interface. Each constructor records an opcode,
its operands, or context/control composition. No constructor can replace a
computed tensor with an arbitrary atom, constant, reduction or expected output. -/
inductive LoweringPlan : Expr→Expr→Type where
  | same (a) : LoweringPlan a a
  | symm {a b} : LoweringPlan a b→LoweringPlan b a
  | trans {a b c} : LoweringPlan a b→LoweringPlan b c→LoweringPlan a c
  | add {a b c d} : LoweringPlan a c→LoweringPlan b d→LoweringPlan (.add a b) (.add c d)
  | sub {a b c d} : LoweringPlan a c→LoweringPlan b d→LoweringPlan (.sub a b) (.sub c d)
  | mul {a b c d} : LoweringPlan a c→LoweringPlan b d→LoweringPlan (.mul a b) (.mul c d)
  | div {a b c d} : LoweringPlan a c→LoweringPlan b d→LoweringPlan (.div a b) (.div c d)
  | fma {a b c d e f} : LoweringPlan a d→LoweringPlan b e→LoweringPlan c f→LoweringPlan (.fma a b c) (.fma d e f)
  | unary (op) {a b} : LoweringPlan a b→LoweringPlan (.unary op a) (.unary op b)
  | addSwap (a b) : LoweringPlan (.add a b) (.add b a)
  | mulSwap (a b) : LoweringPlan (.mul a b) (.mul b a)
  | addZeroLeft (a) : LoweringPlan (.add .zero a) a
  | addZeroRight (a) : LoweringPlan (.add a .zero) a
  | divOne (a) : LoweringPlan (.div a .one) a
  | fuse (a b c) : LoweringPlan (.add (.mul a b) c) (.fma a b c)
  | unaryLink (left right) (a) (tags : (left=0 ∧ right=5) ∨ (left=1 ∧ right=6) ∨ (left=2 ∧ right=7)) :
      LoweringPlan (.unary left a) (.unary right a)
  | castErase (op) (a) (tag : op=0 ∨ op=5) : LoweringPlan (.unary op a) a
  | load (kind col) (kindBound : 2≤kind ∧ kind≤4) :
      LoweringPlan (.atom (Nat.pair kind col)) (.unary 3 (.atom (Nat.pair kind col)))
  | literal (kind) (kindBound : kind≤1) :
      LoweringPlan (.atom (Nat.pair kind 0)) (.unary 4 (.atom (Nat.pair kind 0)))
  | zeroLiteral : LoweringPlan .zero (.unary 4 .zero)
  | oneLiteral : LoweringPlan .one (.unary 4 .one)

def LoweringPlan.steps {a b : Expr} : LoweringPlan a b→List LocalStep
  | .same _ => []
  | .symm x => x.steps
  | .trans x y | .add x y | .sub x y | .mul x y | .div x y => x.steps++y.steps
  | .fma x y z => x.steps++y.steps++z.steps
  | .unary _ x => x.steps
  | .addSwap a b => [.addSwap a b]
  | .mulSwap a b => [.mulSwap a b]
  | .addZeroLeft a => [.addZeroLeft a]
  | .addZeroRight a => [.addZeroRight a]
  | .divOne a => [.divOne a]
  | .fuse a b c => [.fuse a b c]
  | .unaryLink l r a _ => [.unaryLink l r a]
  | .castErase op a _ => [.castErase op a]
  | .load k c _ => [.load k c]
  | .literal k _ => [.literal (.atom (Nat.pair k 0))]
  | .zeroLiteral => [.literal .zero]
  | .oneLiteral => [.literal .one]

theorem LoweringPlan.sound (p : Primitives α) (input : Nat→α) {used : LocalStep→Prop}
    (laws : LocalLaws p input used) {a b : Expr} (plan : LoweringPlan a b) :
    (∀ step, step∈plan.steps→used step) → eval p input a=eval p input b := by
  induction plan with
  | same a => intro _; rfl
  | symm x hx => intro h; exact (hx h).symm
  | trans x y hx hy =>
    intro h
    exact (hx (fun st hs => h st (List.mem_append_left _ hs))).trans
      (hy (fun st hs => h st (List.mem_append_right _ hs)))
  | add x y hx hy | sub x y hx hy | mul x y hx hy | div x y hx hy =>
    intro h
    simp only [eval,hx (fun st hs => h st (List.mem_append_left _ hs)),
      hy (fun st hs => h st (List.mem_append_right _ hs))]
  | fma x y z hx hy hz =>
    intro h
    simp only [eval,hx (fun st hs => h st (by simp [LoweringPlan.steps,hs])),
      hy (fun st hs => h st (by simp [LoweringPlan.steps,hs])),
      hz (fun st hs => h st (by simp [LoweringPlan.steps,hs]))]
  | unary op x hx => intro h; simp only [eval,hx h]
  | addSwap a b => intro h; exact laws.addSwap a b (h _ (by simp [steps]))
  | mulSwap a b => intro h; exact laws.mulSwap a b (h _ (by simp [steps]))
  | addZeroLeft a => intro h; exact laws.addZeroLeft a (h _ (by simp [steps]))
  | addZeroRight a => intro h; exact laws.addZeroRight a (h _ (by simp [steps]))
  | divOne a => intro h; exact laws.divOne a (h _ (by simp [steps]))
  | fuse a b c => intro h; exact laws.fuse a b c (h _ (by simp [steps]))
  | unaryLink l r a tags => intro h; exact laws.unaryLink l r a (h _ (by simp [steps]))
  | castErase op a tag => intro h; exact laws.castErase op a (h _ (by simp [steps]))
  | load k c hb => intro h; exact laws.load k c (h _ (by simp [steps]))
  | literal k hb => intro h; exact laws.literal (.atom (Nat.pair k 0)) (h _ (by simp [steps]))
  | zeroLiteral => intro h; exact laws.literal .zero (h _ (by simp [steps]))
  | oneLiteral => intro h; exact laws.literal .one (h _ (by simp [steps]))

#axiomsClean LoweringPlan.sound
end VeriTile.Bench.Repairs.LayerNormWelfold.Native
