import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardGrid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
noncomputable section

abbrev Word := BitVec 32

/-- Every numerical field is a scalar operation. There is deliberately no
list, row, reduction, kernel, tensor, state or collective numerical port. -/
structure Numerics where
  literal : ℝ → Word
  natural : Nat → Word
  negativeInfinity : Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  neg : Word → Word
  fma : Word → Word → Word → Word
  exp : Word → Word
  log : Word → Word
  maximum : Word → Word → Word
  hostAdd : Word → Word → Word
  hostSquare : Word → Word
  hostScale : ℝ → Word → Word
  hostSub : Word → Word → Word
  hostExp : Word → Word
  hostLog : Word → Word
  hostMaximum : Word → Word → Word
  rankAdd : Word → Word → Word

structure Obligations (native reference : Numerics) : Prop where
  literal : ∀ x, native.literal x=reference.literal x
  natural : ∀ x, native.natural x=reference.natural x
  negativeInfinity : native.negativeInfinity=reference.negativeInfinity
  add : ∀ x y, native.add x y=reference.add x y
  sub : ∀ x y, native.sub x y=reference.sub x y
  mul : ∀ x y, native.mul x y=reference.mul x y
  div : ∀ x y, native.div x y=reference.div x y
  neg : ∀ x, native.neg x=reference.neg x
  fma : ∀ x y z, native.fma x y z=reference.fma x y z
  exp : ∀ x, native.exp x=reference.exp x
  log : ∀ x, native.log x=reference.log x
  maximum : ∀ x y, native.maximum x y=reference.maximum x y
  hostAdd : ∀ x y, native.hostAdd x y=reference.hostAdd x y
  hostSquare : ∀ x, native.hostSquare x=reference.hostSquare x
  hostScale : ∀ s x, native.hostScale s x=reference.hostScale s x
  hostSub : ∀ x y, native.hostSub x y=reference.hostSub x y
  hostExp : ∀ x, native.hostExp x=reference.hostExp x
  hostLog : ∀ x, native.hostLog x=reference.hostLog x
  hostMaximum : ∀ x y, native.hostMaximum x y=reference.hostMaximum x y
  rankAdd : ∀ x y, native.rankAdd x y=reference.rankAdd x y

inductive ScalarBinary where
  | add | sub | mul | div | maximum | hostAdd | hostSub | hostMaximum | rankAdd
  deriving DecidableEq
inductive ScalarUnary where
  | neg | exp | log | hostSquare | hostExp | hostLog | infinityToZero
  deriving DecidableEq

def ScalarBinary.run (n : Numerics) : ScalarBinary → Word → Word → Word
  | .add => n.add | .sub => n.sub | .mul => n.mul | .div => n.div
  | .maximum => n.maximum | .hostAdd => n.hostAdd | .hostSub => n.hostSub
  | .hostMaximum => n.hostMaximum | .rankAdd => n.rankAdd

/-- Torch logsumexp masks both signs of infinity in the maximum before
subtracting. This exact bit test does not mask NaNs, including payload NaNs. -/
def infinityToZero (x : Word) : Word :=
  if x &&& 0x7fffffff = 0x7f800000 then 0 else x

def ScalarUnary.run (n : Numerics) : ScalarUnary → Word → Word
  | .neg => n.neg | .exp => n.exp | .log => n.log
  | .hostSquare => n.hostSquare | .hostExp => n.hostExp | .hostLog => n.hostLog
  | .infinityToZero => Floating.infinityToZero

/-- A communication edge records the source and destination of one word.
It changes neither its bits nor the order of the enclosing scalar operands. -/
inductive Transfer where
  | shuffle (sourceLane destinationLane offset : Nat)
  | shared (writeEpoch readEpoch address sourceThread destinationThread : Nat)
  | global (writeEpoch readEpoch address sourceCTA destinationCTA : Nat)
  | rank (channel sequence sourceRank destinationRank : Nat)
  deriving DecidableEq

/-- A finite scalar graph. All inputs are indexed and all arities are at
most three. No constructor can request a numerical reduction implementation. -/
inductive ScalarGraph (width : Nat) where
  | input (index : Fin width)
  | raw (bits : Word)
  | literal (value : ℝ)
  | natural (value : Nat)
  | negativeInfinity
  | binary (op : ScalarBinary) (left right : ScalarGraph width)
  | unary (op : ScalarUnary) (value : ScalarGraph width)
  | fma (left right addend : ScalarGraph width)
  | hostScale (scale : ℝ) (value : ScalarGraph width)
  | transfer (edge : Transfer) (value : ScalarGraph width)

def ScalarGraph.eval (n : Numerics) (xs : Fin width → Word) : ScalarGraph width → Word
  | .input i => xs i
  | .raw x => x
  | .literal x => n.literal x
  | .natural x => n.natural x
  | .negativeInfinity => n.negativeInfinity
  | .binary op a b => op.run n (a.eval n xs) (b.eval n xs)
  | .unary op a => op.run n (a.eval n xs)
  | .fma a b c => n.fma (a.eval n xs) (b.eval n xs) (c.eval n xs)
  | .hostScale s a => n.hostScale s (a.eval n xs)
  | .transfer _ a => a.eval n xs

def ScalarGraph.bind (xs : Fin width → ScalarGraph other) : ScalarGraph width → ScalarGraph other
  | .input i => xs i
  | .raw x => .raw x
  | .literal x => .literal x
  | .natural x => .natural x
  | .negativeInfinity => .negativeInfinity
  | .binary op a b => .binary op (a.bind xs) (b.bind xs)
  | .unary op a => .unary op (a.bind xs)
  | .fma a b c => .fma (a.bind xs) (b.bind xs) (c.bind xs)
  | .hostScale s a => .hostScale s (a.bind xs)
  | .transfer e a => .transfer e (a.bind xs)

theorem ScalarGraph.eval_bind (n : Numerics) (values : Fin other → Word)
    (xs : Fin width → ScalarGraph other) (g : ScalarGraph width) :
    (g.bind xs).eval n values=g.eval n (fun i => (xs i).eval n values) := by
  induction g <;> simp_all [bind,eval]

theorem scalar_binary_equiv (h : Obligations n r) (op : ScalarBinary) (x y : Word) :
    op.run n x y=op.run r x y := by
  cases op <;> simp only [ScalarBinary.run,h.add,h.sub,h.mul,h.div,h.maximum,
    h.hostAdd,h.hostSub,h.hostMaximum,h.rankAdd]

theorem scalar_unary_equiv (h : Obligations n r) (op : ScalarUnary) (x : Word) :
    op.run n x=op.run r x := by
  cases op <;> simp only [ScalarUnary.run,h.neg,h.exp,h.log,h.hostSquare,h.hostExp,h.hostLog]

/-- Structural composition of scalar identities. The two numerical records
are never equated, and there is no whole-list or whole-graph premise. -/
theorem ScalarGraph.equiv (h : Obligations n r) (xs ys : Fin width → Word)
    (inputs : ∀ i, xs i=ys i) (g : ScalarGraph width) :
    g.eval n xs=g.eval r ys := by
  induction g with
  | input i => exact inputs i
  | raw => rfl
  | literal x => exact h.literal x
  | natural x => exact h.natural x
  | negativeInfinity => exact h.negativeInfinity
  | binary op a b ia ib => simp only [eval,ia,ib,scalar_binary_equiv h]
  | unary op a ia => simp only [eval,ia,scalar_unary_equiv h]
  | fma a b c ia ib ic => simp only [eval,ia,ib,ic,h.fma]
  | hostScale s a ia => simp only [eval,ia,h.hostScale]
  | transfer e a ia => exact ia

#axiomsClean ScalarGraph.eval_bind
#axiomsClean ScalarGraph.equiv
#specNonCircular Obligations avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
