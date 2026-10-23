import bench.repairs.tritonbench_g.fused_recurrent_retention.Kernels
import VeriTile.Triton.Float.Structural

/-! Local compiler arithmetic model for the unchanged source quotation.
Tags record static producer information, never predicates on numerical words.
They preserve operands across the two multiply/add contraction sites and the
four product reductions. In particular reverse-loop decay is materialized
before the next iteration. This is an explicit compiler-model boundary, not
an asserted equivalence of the whole quotation with an event interpreter. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

inductive Kind where
  | plain | one | exponential | beta | scale | scaledQuery
  | forwardProduct | reverseProduct | forwardState | reverseState | reverseCarry
  | outputDot | queryDot | reverseDot | product
  deriving DecidableEq, Repr

structure Node (α : Type) where
  value : α
  kind : Kind := .plain
  operands : Option (α × α) := none

instance [Inhabited α] : Inhabited (Node α) := ⟨⟨default,.plain,none⟩⟩
def atom (x : α) : Node α := ⟨x,.plain,none⟩
def tagged (kind : Kind) (x : α) : Node α := ⟨x,kind,none⟩
def product (a : Arithmetic α) (kind : Kind) (x y : α) : Node α :=
  ⟨a.mul x y,kind,some (x,y)⟩

def multiply (a : Arithmetic α) (x y : Node α) : Node α :=
  if x.kind = .beta then product a .forwardProduct x.value y.value
  else if y.kind = .beta then tagged .reverseCarry (a.mul x.value y.value)
  else if y.kind = .scale then tagged .scaledQuery (a.mul x.value y.value)
  else if x.kind = .scaledQuery then product a .reverseProduct x.value y.value
  else if x.kind = .forwardState then
    product a (if y.kind = .scaledQuery then .outputDot else .queryDot) x.value y.value
  else if x.kind = .reverseState then product a .reverseDot x.value y.value
  else product a .product x.value y.value

def add (a : Arithmetic α) (x y : Node α) : Node α :=
  match x.kind,x.operands,y.kind,y.operands with
  | .forwardProduct,some (l,r),_,_ => tagged .forwardState (a.fma l r y.value)
  | _,_,.reverseProduct,some (l,r) => tagged .reverseState (a.fma l r x.value)
  | _,_,_,_ => atom (a.add x.value y.value)

def subtract (a : Arithmetic α) (x y : Node α) : Node α :=
  tagged (if x.kind = .one ∧ y.kind = .exponential then .beta else .plain)
    (a.sub x.value y.value)

def literal (a : Arithmetic α) (x : ℝ) : Node α :=
  if x=1 then tagged .one a.one else atom (if x= -5 then a.negFive else a.zero)

def choosePlan (p : Plans) (axno : Nat) : Kind → DotPlan
  | .outputDot => p.output
  | .queryDot => p.dq
  | .reverseDot => if axno=1 then p.dk else p.dv
  | _ => .product 0

def reduceNodes (a : Arithmetic α) (p : Plans) (axno : Nat) (nodes : Nat → Node α) : Node α :=
  let operands := fun i => (nodes i).operands.getD ((nodes i).value,a.one)
  atom ((choosePlan p axno (nodes 0).kind).eval a
    (fun i => (operands i).1) (fun i => (operands i).2))

def sumValues (a : Arithmetic α) (p : Plans) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values (Node α) .real shape) :
    Values (Node α) .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => reduceNodes a p ax.val (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.insertAxisIndex shape ax out ⟨j,h⟩) else atom a.zero)
  · exact fun out => reduceNodes a p ax.val (fun j =>
      if h : j<TileShape.axisDim shape ax then
        values (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩) else atom a.zero)

/-- Unused operations have explicit total defaults. The execution lemmas use
only literals 0, 1 and negative 5, unsigned head conversion, fp32 identity loads/casts,
exp2, sub, multiply, add and the four reductions above. -/
def algebra (a : Arithmetic α) (p : Plans) : Algebra (Node α) where
  literal := fun _ _ => literal a
  negInf := atom a.zero
  binary := fun _ _ op x y => match op with
    | .add => add a x y | .sub => subtract a x y | .mul => multiply a x y
    | _ => atom a.zero
  unary := fun _ op x => match op with
    | .exp2 => tagged .exponential (a.exp2 x.value)
    | _ => atom a.zero
  cast := fun _ _ _ x => atom x.value
  fromNat := fun _ n => atom (a.fromNat n)
  fromInt := fun _ n => atom (a.fromNat n.toNat)
  fp32Bits := fun _ => atom a.zero
  fp32Load := fun x => atom x.value
  reduceMax := fun _ {_} _ _ _ _ => atom a.zero
  reduceSum := fun _ => sumValues a p

@[simp] theorem forward_step (a : Arithmetic α) (beta : α) (old : Node α) (key value : α) :
    add a (multiply a (tagged .beta beta) old)
      (multiply a (atom key) (atom value)) =
      tagged .forwardState (a.fma beta old.value (a.mul key value)) := by
  simp [add,multiply,tagged,product,atom]

@[simp] theorem scaled_query (a : Arithmetic α) (q scale : α) :
    multiply a (atom q) (tagged .scale scale) = tagged .scaledQuery (a.mul q scale) := by
  simp [multiply,atom,tagged]

@[simp] theorem reverse_step (a : Arithmetic α) (old : Node α) (q scale dout : α)
    (notForward : old.kind ≠ .forwardProduct) :
    add a old (multiply a (multiply a (atom q) (tagged .scale scale)) (atom dout)) =
      tagged .reverseState (a.fma (a.mul q scale) dout old.value) := by
  rw [scaled_query]
  cases old with
  | mk value kind operands =>
    cases kind <;> simp_all [add,multiply,tagged,product,atom]

@[simp] theorem reverse_decay (a : Arithmetic α) (grad beta : α) :
    multiply a (tagged .reverseState grad) (tagged .beta beta) =
      tagged .reverseCarry (a.mul grad beta) := by
  simp [multiply,tagged]

@[simp] theorem forward_product (a : Arithmetic α) (h q : α) :
    multiply a (tagged .forwardState h) (tagged .scaledQuery q) =
      product a .outputDot h q := by simp [multiply,tagged]

@[simp] theorem query_product (a : Arithmetic α) (h dout : α) :
    multiply a (tagged .forwardState h) (atom dout) =
      product a .queryDot h dout := by simp [multiply,tagged,atom]

@[simp] theorem reverse_product (a : Arithmetic α) (h x : α) :
    multiply a (tagged .reverseState h) (atom x) =
      product a .reverseDot h x := by simp [multiply,tagged,atom]

@[simp] theorem reduction_products (a : Arithmetic α) (p : Plans) (axno : Nat)
    (kind : Kind) (x y : Nat → α) :
    reduceNodes a p axno (fun i => product a kind (x i) (y i)) =
      atom ((choosePlan p axno kind).eval a x y) := by rfl

#axiomsClean forward_step
#axiomsClean reverse_step
#axiomsClean reverse_decay
#axiomsClean reduction_products
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Staged
