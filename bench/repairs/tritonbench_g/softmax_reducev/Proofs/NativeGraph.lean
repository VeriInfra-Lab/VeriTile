import bench.repairs.tritonbench_g.softmax_reducev.Proofs.WholeHost
namespace VeriTile.Bench.Repairs.SoftmaxReduceV.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

abbrev Word := BitVec 32

/-- Floating instructions, including the exact default-fusion FMA. -/
structure Instructions where
  (maxNum subRN mulRN addRN divFull : Word → Word → Word)
  fmaRN : Word → Word → Word → Word
  exp2Approx : Word → Word

inductive Expr where
  | word (bits : Word)
  | literal (value : ℝ)
  | sub (a b : Expr)
  | product (a b : Expr)
  | add (a b : Expr)
  | maximum (a b : Expr)
  | divide (a b : Expr)
  | fma (a b c : Expr)
  | exp2 (a : Expr)
  | unused

instance : Inhabited Expr := ⟨.word 0⟩
abbrev Node := Nat → Expr

def atom (w : Word) : Node := fun _ => .word w

/-- These choices are instruction selections at an ordered add node. They
never change the order of its children or supply a floating result. -/
inductive Fusion where
  | separate | left | right | both
  deriving DecidableEq

def add (mode : Fusion) (a b : Expr) : Expr :=
  match mode,a,b with
  | .left,.product x y,z | .both,.product x y,z => .fma x y z
  | .right,z,.product x y | .both,z,.product x y => .fma x y z
  | _,x,y => .add x y

/-- Each leaf retains its logical source index and physical replica. A shuffle
or shared-memory conversion changes which replica is read, not its value. -/
inductive Tree (n : Nat) where
  | input (i : Fin n) (replica : Nat)
  | padding (word : Word)
  | join (fusion : Fusion) (a b : Tree n)

def Tree.indices : Tree n → List (Fin n)
  | .input i _ => [i]
  | .padding _ => []
  | .join _ a b => a.indices++b.indices

structure OrderedTree (n : Nat) where
  tree : Tree n
  complete : tree.indices.Perm (List.finRange n)

/-- Only structural profile data: source-index-complete ordered trees,
physical replicas, FMA sites, and the store's source lane. No numerical
operation, reduction result, or final-output field exists in this record. -/
structure Plan where
  sumTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Nat → OrderedTree (TileShape.axisDim shape ax)
  maxTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Nat → OrderedTree (TileShape.axisDim shape ax)
  storeReplica : Nat → Nat → Nat

def inputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → Node) (i : TileIndex (TileShape.reduceShape shape ax keep)) :
    Fin (TileShape.axisDim shape ax) → Node :=
  match keep with
  | Bool.false => fun k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun k => xs (TileShape.replaceAxisIndex shape ax i k)

def treeExpr (isMax : Bool) (xs : Fin n → Node) : Tree n → Expr
  | .input i replica => xs i replica
  | .padding w => .word w
  | .join fusion a b => if isMax then .maximum (treeExpr isMax xs a) (treeExpr isMax xs b)
    else add fusion (treeExpr isMax xs a) (treeExpr isMax xs b)

def reduction (plan : Plan) (isMax : Bool) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (xs : TileIndex shape → Node)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Node := fun replica =>
  treeExpr isMax (inputs ax keep xs i)
    (if isMax then (plan.maxTree ax keep i replica).tree else (plan.sumTree ax keep i replica).tree)

/-- Natural exponentiation retains its actual multiply by the FP32 log2(e)
word, then ex2.approx. No exact-real exp identity enters native execution. -/
def exponential (x : Expr) : Expr := .exp2 (.product x (.word 0x3fb8aa3b))

def binary (op : Binary) (a b : Expr) : Expr := match op with
  | .add => add .left a b
  | .sub => .sub a b
  | .mul => .product a b
  | .div => .divide a b
  | .max => .maximum a b
  | .pow => .unused

def algebra (plan : Plan) : Algebra Node where
  literal := fun _ _ r _ => .literal r
  negInf := atom 0xff800000
  binary := fun _ _ op a b replica => binary op (a replica) (b replica)
  unary := fun _ op x replica => match op with
    | .exp => exponential (x replica)
    | _ => .unused
  cast := fun _ _ _ x => x
  fromNat := fun _ _ _ => .unused
  fromInt := fun _ _ _ => .unused
  fp32Bits := fun w => atom w.bits
  fp32Load := id
  reduceSum := fun _ {_} => reduction plan Bool.false
  reduceMax := fun _ {_} => reduction plan Bool.true

/-- The source only uses real zero literals. Other literal/operation cases
are explicitly unreachable for this source, not additional numerical ports. -/
def eval (ops : Instructions) : Expr → Word
  | .word w => w
  | .literal r => if r=0 then 0 else 0xffc00000
  | .sub a b => ops.subRN (eval ops a) (eval ops b)
  | .product a b => ops.mulRN (eval ops a) (eval ops b)
  | .add a b => ops.addRN (eval ops a) (eval ops b)
  | .maximum a b => ops.maxNum (eval ops a) (eval ops b)
  | .divide a b => ops.divFull (eval ops a) (eval ops b)
  | .fma a b c => ops.fmaRN (eval ops a) (eval ops b) (eval ops c)
  | .exp2 a => ops.exp2Approx (eval ops a)
  | .unused => 0xffc00000

def nodeInput (x : Host.Input Word) : Host.Input Node :=
  ⟨fun a => atom (x.logics a),fun a => atom (x.values a),x.starts,x.counts,x.pages⟩

def row (plan : Plan) (ops : Instructions) (c : Host.Call) (x : Host.Input Word)
    (j : Host.Job) (d : Nat) : Word :=
  eval ops (Host.row (algebra plan) c (nodeInput x) j d (plan.storeReplica c.dimensions d))

def observe (plan : Plan) (ops : Instructions) (c : Host.Call) (j : Host.Job)
    (t : State Node) (a : Nat) : Word :=
  eval ops ((t.mem "Out" a).read .fp32 (plan.storeReplica c.dimensions (a-Host.address c j 0)))

def evaluate (plan : Plan) (ops : Instructions) (c : Host.Call) (x : Host.Input Word)
    (j : Host.Job) (out : Nat → Word) : Option (Nat → Word) :=
  (FP.Structural.exec (algebra plan) (FPValue.source (Host.parameters c))
    (Host.launch c (nodeInput x) (fun a => atom (out a)) j)).map (observe plan ops c j)

/-- The complete typed source produces the native scalar graph. Only scalar
instruction operations are ports; every reduction is an explicit tree with
complete indexed leaves, and both local reduction and update FMA are retained. -/
theorem source_law (plan : Plan) (ops : Instructions) (c : Host.Call) (x : Host.Input Word)
    (j : Host.Job) (s t : Nat → Word) (run : evaluate plan ops c x j s=some t) :
    (∀ d, d < c.dimensions → t (Host.address c j d)=row plan ops c x j d) ∧ Host.Frame c [j] s t := by
  obtain ⟨u,exec,post⟩ := FPValue.correctness (algebra plan) (Host.parameters c)
    (Host.launch c (nodeInput x) (fun a => atom (s a)) j) (by change 0 < 64;decide)
    (Host.source_injective c (nodeInput x) (fun a => atom (s a)) j)
  have result : observe plan ops c j u=t := by simpa [evaluate,exec] using run
  subst t
  constructor
  · intro d hd
    have value := post.1 ((⟨d,hd⟩ : Fin c.dimensions),())
    rw [Host.source_address] at value
    change u.mem "Out" (Host.address c j d)=_ at value
    dsimp [observe]
    rw [value]
    simp only [Cell.read_same,Host.address,Nat.add_zero,Nat.add_sub_cancel_left]
    simp only [row,Host.row,dif_pos hd]
    rw [Host.accumulator_independent (algebra plan) c (nodeInput x) (fun a => atom (s a)) (fun _ => default) j]
    rfl
  · intro a outside
    have eq := post.2 "Out" a (Or.inr (by
      intro d eq
      apply outside j (by simp)
      exact ⟨d.1.val,d.1.isLt,eq.trans (Host.source_address c (nodeInput x) (fun a => atom (s a)) j d)⟩))
    simp only [observe,eq]
    simp [Host.launch,atom,eval]

/-- Local instruction conformance only. There is deliberately no whole-exp,
whole-reduction, online-loop, source-execution, or final-output callback. -/
structure InstructionAgreement (a b : Instructions) : Prop where
  maxNum : ∀ x y, a.maxNum x y=b.maxNum x y
  subRN : ∀ x y, a.subRN x y=b.subRN x y
  mulRN : ∀ x y, a.mulRN x y=b.mulRN x y
  addRN : ∀ x y, a.addRN x y=b.addRN x y
  divFull : ∀ x y, a.divFull x y=b.divFull x y
  fmaRN : ∀ x y z, a.fmaRN x y z=b.fmaRN x y z
  exp2Approx : ∀ x, a.exp2Approx x=b.exp2Approx x

theorem instruction_equality (a b : Instructions) (h : InstructionAgreement a b) : a=b := by
  cases a; cases b
  have h1 := funext fun x => funext (h.maxNum x)
  have h2 := funext fun x => funext (h.subRN x)
  have h3 := funext fun x => funext (h.mulRN x)
  have h4 := funext fun x => funext (h.addRN x)
  have h5 := funext fun x => funext (h.divFull x)
  have h6 := funext fun x => funext fun y => funext (h.fmaRN x y)
  have h7 := funext h.exp2Approx
  simp_all

theorem row_agreement (plan : Plan) (a b : Instructions) (h : InstructionAgreement a b)
    (c : Host.Call) (x : Host.Input Word) (j : Host.Job) (d : Nat) :
    row plan a c x j d=row plan b c x j d := by rw [instruction_equality a b h]

#axiomsClean source_law
#axiomsClean instruction_equality
#axiomsClean row_agreement
end
end VeriTile.Bench.Repairs.SoftmaxReduceV.NativeGraph
