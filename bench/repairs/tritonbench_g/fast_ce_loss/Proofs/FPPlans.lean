import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.API

namespace VeriTile.Bench.Repairs.FastCeLoss.Floating
open Source
noncomputable section
set_option maxHeartbeats 4000000

abbrev Word := BitVec 32

inductive Unary where
  | load | cast | exp | log | tanh | hostExp | hostLog | hostFiniteMax
inductive Binary where
  | add | sub | mul | div | maximum | hostAdd | hostSub | hostDiv | upstreamDiv | hostMaximum

/-- The source graph retains casts, storage loads, Torch arithmetic and each
source reduction site. It is independent of the mathematical specification. -/
inductive Expr where
  | raw (word : Word)
  | literal (value : ℝ)
  | natural (value : Nat)
  | pythonInt (value : Int)
  | pythonFloat (bits : BitVec 64)
  | negInf
  | unary (operation : Unary) (value : Expr)
  | binary (operation : Binary) (left right : Expr)
  | sum (width : Nat) (values : Fin width → Expr)
  | maximum (width : Nat) (values : Fin width → Expr)
  | hostSum (width : Nat) (values : Fin width → Expr)
  | hostLSE (width : Nat) (values : Fin width → Expr)

instance : Inhabited Expr := ⟨.raw 0⟩

def arithmetic : Arithmetic Expr where
  literal := .literal
  natural := .natural
  negInf := .negInf
  load := .unary .load
  cast := .unary .cast
  add := .binary .add
  sub := .binary .sub
  mul := .binary .mul
  div := .binary .div
  exp := .unary .exp
  log := .unary .log
  tanh := .unary .tanh
  maximum := fun {n} xs => .maximum n xs
  sum := fun {n} xs => .sum n xs
  hostSum := fun {n} xs => .hostSum n xs
  hostLSE := fun {n} xs => .hostLSE n xs
  hostAdd := .binary .hostAdd
  hostDiv := .binary .hostDiv
  upstreamDiv := .binary .upstreamDiv

/-- All ports are scalar FP32 operations. hostFiniteMax is Torch's scalar
abs(max)==inf ? 0 : max normalization, including its actual NaN behavior. -/
inductive Primitive : Nat → Type where
  | literal (x : ℝ) : Primitive 0
  | natural (x : Nat) : Primitive 0
  | pythonInt (x : Int) : Primitive 0
  | pythonFloat (bits : BitVec 64) : Primitive 0
  | negInf : Primitive 0
  | unary (op : Unary) : Primitive 1
  | binary (op : Binary) : Primitive 2
  | fma : Primitive 3
  | fms : Primitive 3
  | fnma : Primitive 3

/-- Loads and same-format casts are fixed bit identities; -infinity is fixed
IEEE FP32 bits. They are structural facts and never numerical deferrals. -/
def Primitive.computational : {arity : Nat} → Primitive arity → Prop
  | _,.unary .load => False
  | _,.unary .cast => False
  | _,.negInf => False
  | _,_ => True

structure Numerics where
  scalar : {arity : Nat} → Primitive arity → (Fin arity → Word) → Word

def Numerics.evaluate (n : Numerics) : {arity : Nat} → Primitive arity → (Fin arity → Word) → Word
  | _,.unary .load,xs => xs 0
  | _,.unary .cast,xs => xs 0
  | _,.negInf,_ => BitVec.ofNat 32 0xff800000
  | _,op,xs => n.scalar op xs

/-- Actual ordered scalar instruction graph. Reductions are expanded into
binary operations; there is no whole-sum or whole-LSE numerical primitive. -/
inductive Graph where
  | raw (word : Word)
  | op {arity : Nat} (operation : Primitive arity) (operands : Fin arity → Graph)

def Graph.eval (n : Numerics) : Graph → Word
  | .raw word => word
  | .op operation xs => n.evaluate operation (fun i => (xs i).eval n)

abbrev Domain := {arity : Nat} → Primitive arity → (Fin arity → Word) → Prop

/-- Deferred equality only on explicitly selected local operand tuples. -/
structure LocalPorts (native reference : Numerics) (domain : Domain) : Prop where
  scalar : ∀ {arity} (op : Primitive arity) (xs : Fin arity → Word),
    op.computational → domain op xs → native.scalar op xs=reference.scalar op xs

theorem Numerics.equivalent (native reference : Numerics) (domain : Domain)
    (ports : LocalPorts native reference domain) {arity : Nat} (op : Primitive arity)
    (xs : Fin arity → Word) (admitted : op.computational → domain op xs) :
    native.evaluate op xs=reference.evaluate op xs := by
  cases op with
  | unary operation =>
    cases operation <;> first | rfl | exact ports.scalar _ xs (by trivial) (admitted (by trivial))
  | _ => first | rfl | exact ports.scalar _ xs (by trivial) (admitted (by trivial))

def Graph.Admitted (n : Numerics) (domain : Domain) : Graph → Prop
  | .raw _ => True
  | .op operation xs => (∀ i,(xs i).Admitted n domain) ∧ (operation.computational → domain operation (fun i => (xs i).eval n))

theorem Graph.equivalent (native reference : Numerics) (domain : Domain)
    (ports : LocalPorts native reference domain) (graph : Graph)
    (admitted : graph.Admitted native domain) : graph.eval native=graph.eval reference := by
  induction graph with
  | raw => rfl
  | op op xs ih =>
    change native.evaluate op (fun i => (xs i).eval native)=reference.evaluate op (fun i => (xs i).eval reference)
    rw [Numerics.equivalent native reference domain ports op _ admitted.2]
    congr 1
    funext i
    exact ih i (admitted.1 i)

def nullary (p : Primitive 0) : Graph := .op p Fin.elim0
def unary (p : Unary) (x : Graph) : Graph := .op (.unary p) (fun _ => x)
def binary (p : Binary) (x y : Graph) : Graph := .op (.binary p) ![x,y]
def ternary (p : Primitive 3) (x y z : Graph) : Graph := .op p ![x,y,z]

/-- The selected reduction tree may contain identity leaves and any ordering,
but each actual lane occurs exactly once. This is structural compiler data. -/
inductive Tree (width : Nat) where
  | identity
  | lane (index : Fin width)
  | join (left right : Tree width)

def Tree.lanes {width : Nat} : Tree width → List (Fin width)
  | .identity => []
  | .lane i => [i]
  | .join left right => left.lanes++right.lanes

structure Reduction (width : Nat) where
  tree : Tree width
  covers : tree.lanes.Perm (List.ofFn (fun i : Fin width => i))

def Tree.graph {width : Nat} (op : Binary) (zero : Graph) (values : Fin width → Graph) : Tree width → Graph
  | .identity => zero
  | .lane i => values i
  | .join left right => binary op (left.graph op zero values) (right.graph op zero values)

/-- Dynamic source identity belongs to the actual installed request: path and
visit distinguish two occurrences with the same erased expression. -/
structure Site where
  request : Nat
  astPath : List Nat
  visit : Nat

/-- Selected plans keep the concrete source expression as their index. The
fusion constructors preserve the corresponding FMA operand order; none equate
FMA with separate multiplication/addition. -/
inductive Plan : Expr → Type where
  | raw (word : Word) : Plan (.raw word)
  | literal (site : Site) (x : ℝ) : Plan (.literal x)
  | natural (site : Site) (x : Nat) : Plan (.natural x)
  | pythonInt (site : Site) (x : Int) : Plan (.pythonInt x)
  | pythonFloat (site : Site) (bits : BitVec 64) : Plan (.pythonFloat bits)
  | negInf (site : Site) : Plan .negInf
  | unary (site : Site) (op : Unary) {x : Expr} (value : Plan x) : Plan (.unary op x)
  | binary (site : Site) (op : Binary) {x y : Expr} (left : Plan x) (right : Plan y) : Plan (.binary op x y)
  | mulSwap (site : Site) {x y : Expr} (left : Plan x) (right : Plan y) : Plan (.binary .mul x y)
  | fma (site : Site) (swapped : Bool) {x y z : Expr} (left : Plan x) (right : Plan y) (addend : Plan z) :
      Plan (.binary .add (.binary .mul x y) z)
  | fmaRight (site : Site) (swapped : Bool) {x y z : Expr} (addend : Plan z) (left : Plan x) (right : Plan y) :
      Plan (.binary .add z (.binary .mul x y))
  | fms (site : Site) (swapped : Bool) {x y z : Expr} (left : Plan x) (right : Plan y) (subtrahend : Plan z) :
      Plan (.binary .sub (.binary .mul x y) z)
  | fnma (site : Site) (swapped : Bool) {x y z : Expr} (minuend : Plan z) (left : Plan x) (right : Plan y) :
      Plan (.binary .sub z (.binary .mul x y))
  /-- Actual contraction through an explicit same-FP32 cast. The source cast
  remains in the indexed expression; this is a selected fused lowering, not
  a claim that separate rounded mul/cast/sub equals FMA. -/
  | fmaCast (site castSite : Site) (swapped : Bool) {x y z : Expr} (left : Plan x) (right : Plan y) (addend : Plan z) :
      Plan (.binary .add (.unary .cast (.binary .mul x y)) z)
  | fmaRightCast (site castSite : Site) (swapped : Bool) {x y z : Expr} (addend : Plan z) (left : Plan x) (right : Plan y) :
      Plan (.binary .add z (.unary .cast (.binary .mul x y)))
  | fmsCast (site castSite : Site) (swapped : Bool) {x y z : Expr} (left : Plan x) (right : Plan y) (subtrahend : Plan z) :
      Plan (.binary .sub (.unary .cast (.binary .mul x y)) z)
  | fnmaCast (site castSite : Site) (swapped : Bool) {x y z : Expr} (minuend : Plan z) (left : Plan x) (right : Plan y) :
      Plan (.binary .sub z (.unary .cast (.binary .mul x y)))
  | sum (site : Site) {width : Nat} {xs : Fin width → Expr} (tree : Reduction width)
      (values : ∀ i,Plan (xs i)) : Plan (.sum width xs)
  | maximum (site : Site) {width : Nat} {xs : Fin width → Expr} (tree : Reduction width)
      (values : ∀ i,Plan (xs i)) : Plan (.maximum width xs)
  | hostSum (site : Site) {width : Nat} {xs : Fin width → Expr} (tree : Reduction width)
      (values : ∀ i,Plan (xs i)) : Plan (.hostSum width xs)
  | hostLSE (site : Site) {width : Nat} {xs : Fin width → Expr}
      (maximum sum : Reduction width) (values : ∀ i,Plan (xs i)) : Plan (.hostLSE width xs)

def Plan.graph : {expression : Expr} → Plan expression → Graph
  | _,.raw x => .raw x
  | _,.literal _ x => nullary (.literal x)
  | _,.natural _ x => nullary (.natural x)
  | _,.pythonInt _ x => nullary (.pythonInt x)
  | _,.pythonFloat _ bits => nullary (.pythonFloat bits)
  | _,.negInf _ => nullary .negInf
  | _,.unary _ op p => Floating.unary op p.graph
  | _,.binary _ op p q => Floating.binary op p.graph q.graph
  | _,.mulSwap _ p q => Floating.binary .mul q.graph p.graph
  | _,.fma _ swapped p q r => if swapped then ternary .fma q.graph p.graph r.graph else ternary .fma p.graph q.graph r.graph
  | _,.fmaRight _ swapped r p q => if swapped then ternary .fma q.graph p.graph r.graph else ternary .fma p.graph q.graph r.graph
  | _,.fms _ swapped p q r => if swapped then ternary .fms q.graph p.graph r.graph else ternary .fms p.graph q.graph r.graph
  | _,.fnma _ swapped r p q => if swapped then ternary .fnma q.graph p.graph r.graph else ternary .fnma p.graph q.graph r.graph
  | _,.fmaCast _ _ swapped p q r => if swapped then ternary .fma q.graph p.graph r.graph else ternary .fma p.graph q.graph r.graph
  | _,.fmaRightCast _ _ swapped r p q => if swapped then ternary .fma q.graph p.graph r.graph else ternary .fma p.graph q.graph r.graph
  | _,.fmsCast _ _ swapped p q r => if swapped then ternary .fms q.graph p.graph r.graph else ternary .fms p.graph q.graph r.graph
  | _,.fnmaCast _ _ swapped r p q => if swapped then ternary .fnma q.graph p.graph r.graph else ternary .fnma p.graph q.graph r.graph
  | _,.sum _ tree ps => tree.tree.graph .add (nullary (.literal 0)) (fun i => (ps i).graph)
  | _,.maximum _ tree ps => tree.tree.graph .maximum (nullary .negInf) (fun i => (ps i).graph)
  | _,.hostSum _ tree ps => tree.tree.graph .hostAdd (nullary (.literal 0)) (fun i => (ps i).graph)
  | _,.hostLSE _ maxTree sumTree ps =>
    let m := Floating.unary .hostFiniteMax (maxTree.tree.graph .hostMaximum (nullary .negInf) (fun i => (ps i).graph))
    let terms := fun i => Floating.unary .hostExp (Floating.binary .hostSub (ps i).graph m)
    Floating.binary .hostAdd (Floating.unary .hostLog (sumTree.tree.graph .hostAdd (nullary (.literal 0)) terms)) m

/-- The native and reference interpretations use the same selected tree and
fusion graph. This proof never invokes a whole-output numerical equality. -/
theorem Plan.equivalent {expression : Expr} (plan : Plan expression)
    (native reference : Numerics) (domain : Domain) (ports : LocalPorts native reference domain)
    (admitted : plan.graph.Admitted native domain) :
    plan.graph.eval native=plan.graph.eval reference := Graph.equivalent native reference domain ports _ admitted

structure Node where
  expression : Expr
  plan : Plan expression
  site : Site
  parents : List Nat

/-- Fixed reaching definitions, populated by the one actual execution. -/
structure Origins where
  next : Nat
  nodes : Nat → Option Node
  cells : Address → Option Nat

structure Origins.WellFormed (s : Origins) : Prop where
  earlier : ∀ id node,s.nodes id=some node → id<s.next
  reaching : ∀ cell id,s.cells cell=some id → ∃ node,s.nodes id=some node

def Origins.push (s : Origins) (node : Node) : Origins :=
  {s with next:=s.next+1,nodes:=fun id => if id=s.next then some node else s.nodes id}

def Origins.store (s : Origins) (cell : Address) (id : Nat) : Origins :=
  {s with cells:=fun c => if c=cell then some id else s.cells c}

def Node.transport (node : Node) {expression : Expr} (same : node.expression=expression) : Plan expression :=
  same ▸ node.plan

theorem Node.transport_word (node : Node) {expression : Expr} (same : node.expression=expression)
    (numerics : Numerics) : (node.transport same).graph.eval numerics=node.plan.graph.eval numerics := by
  cases same
  rfl

#axiomsClean Graph.equivalent
#axiomsClean Plan.equivalent
#axiomsClean Node.transport_word
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Floating
