import bench.tritonbench_g.token_attn_reduceV.Proofs.Stores

namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev Word := BitVec 32

/-- Numerical words occur only in leaves. Compiler choices inspect Shape. -/
inductive Shape where
  | leaf (region : RegionName) (address : Nat)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | negInf
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Shape)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Shape)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Shape)
  | fromNat (precision : Option ComputeDType) (n : Nat)
  | fromInt (precision : Option ComputeDType) (n : Int)
  | bits (word : Word)
  | load (a : Shape)
  deriving Inhabited
inductive Term where
  | leaf (region : RegionName) (address : Nat) (word : Word)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | negInf
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Term)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Term)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Term)
  | fromNat (precision : Option ComputeDType) (n : Nat)
  | fromInt (precision : Option ComputeDType) (n : Int)
  | bits (word : Word)
  | load (a : Term)
  deriving Inhabited

def shape : Term → Shape
  | .leaf r o _ => .leaf r o
  | .literal p d x => .literal p d x
  | .negInf => .negInf
  | .binary p d op a b => .binary p d op (shape a) (shape b)
  | .unary p op a => .unary p op (shape a)
  | .cast p src dst a => .cast p src dst (shape a)
  | .fromNat p n => .fromNat p n
  | .fromInt p n => .fromInt p n
  | .bits w => .bits w
  | .load a => .load (shape a)

inductive Tree (n : Nat) where
  | lane (i : Fin n)
  | neutral
  | join (left right : Tree n)
def Tree.leaves : Tree n → List (Option (Fin n))
  | .lane i => [some i]
  | .neutral => [none]
  | .join a b => a.leaves++b.leaves
structure Plan (n : Nat) where
  tree : Tree n
  padding : Nat
  valid : tree.leaves.Perm ((List.finRange n).map some++List.replicate padding none)

def Tree.expand (p : Option ComputeDType) (dtype : FloatDType) (op : Binary) (xs : Fin n → Term) (neutral : Term) : Tree n → Term
  | .lane i => xs i
  | .neutral => neutral
  | .join a b => .binary p dtype op (a.expand p dtype op xs neutral) (b.expand p dtype op xs neutral)

theorem Plan.lane_coverage (p : Plan n) (i : Fin n) : some i∈p.tree.leaves := by
  apply p.valid.mem_iff.mpr
  simp

def Tree.seeded (xs : List (Fin n)) : Tree n := xs.foldr (fun i t => .join (.lane i) t) .neutral

theorem Tree.seeded_leaves (xs : List (Fin n)) : (Tree.seeded xs).leaves=xs.map some++[none] := by
  induction xs with
  | nil => rfl
  | cons x xs ih => change some x::(Tree.seeded xs).leaves=some x::(xs.map some++[none]); rw [ih]

def Plan.seeded (n : Nat) : Plan n :=
  ⟨Tree.seeded (List.finRange n),1,by rw [Tree.seeded_leaves]; exact .refl _⟩

/-- The actual selected scalar sum tree sees layout and expression shapes,
never runtime numeric words. Every token lane is present exactly once. -/
abbrev Reductions := (precision : Option ComputeDType) → (s : TileShape) →
  (ax : Fin s.length) → (keep : Bool) → (out : TileIndex (TileShape.reduceShape s ax keep)) →
  (Fin (TileShape.axisDim s ax) → Shape) → Plan (TileShape.axisDim s ax)

def inputs (s : TileShape) (ax : Fin s.length) (keep : Bool) (xs : Values Term .fp32 s) :
    TileIndex (TileShape.reduceShape s ax keep) → Fin (TileShape.axisDim s ax) → Term :=
  match keep with
  | Bool.false => fun out k => xs (TileShape.insertAxisIndex s ax out k)
  | Bool.true => fun out k => xs (TileShape.replaceAxisIndex s ax out k)

def reduction (R : Reductions) : SumPrimitive Term where
  apply := fun p {s} ax keep xs out =>
    let leaves := inputs s ax keep xs out
    let plan := R p s ax keep out (fun i => shape (leaves i))
    plan.tree.expand p .fp32 .add leaves (.literal p .fp32 0)

def algebra : Algebra Term where
  literal := .literal
  negInf := .negInf
  binary := .binary
  unary := .unary
  cast := .cast
  fromNat := .fromNat
  fromInt := .fromInt
  fp32Bits := fun w => .bits w.bits
  fp32Load := .load
  reduceMax := fun _ {_} _ _ _ _ => .literal none .real 0
  reduceSum := fun _ {_} _ _ _ _ => .literal none .real 0

/-- Only the three reached scalar instructions are deferred. FP32 storage and
same-format casts are identities; unused operation cases are unreachable. No
kernel, tensor, reduction or whole-result instruction exists here. -/
structure Instructions where
  addRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word

inductive Fusion where
  | split | left | right
  deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion

/-- Only an actual adjacent FP32 mul/add pair may contract. A schedule cannot
introduce a new expression, inspect raw words, or assert FMA equals split ops.
Explicit load/cast nodes remain boundaries. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal _ _ x => if x=0 then 0 else 0xffc00000
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .left,some .fp32,.fp32,.binary (some .fp32) .fp32 .mul x y,c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .right,some .fp32,.fp32,c,.binary (some .fp32) .fp32 .mul x y =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | _,_,_,a,b => I.addRN (eval I S a) (eval I S b)
  | .binary _ _ .mul a b => I.mulRN (eval I S a) (eval I S b)
  | .binary _ _ _ _ _ => 0xffc00000
  | .unary _ _ _ => 0xffc00000
  | .cast _ _ _ a => eval I S a
  | .fromNat _ _ => 0xffc00000
  | .fromInt _ _ => 0xffc00000
  | .bits w => w
  | .load a => eval I S a

theorem left_fused (I : Instructions) (S : Schedule) (a b c : Term)
    (h : S.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul (shape a) (shape b)) (shape c))=.left) :
    eval I S (.binary (some .fp32) .real .add (.binary (some .fp32) .real .mul a b) c)=
      I.fmaRN (eval I S a) (eval I S b) (eval I S c) := by simp [eval,shape,h]

theorem right_fused (I : Instructions) (S : Schedule) (a b c : Term)
    (h : S.fusion (.binary (some .fp32) .real .add (shape c)
      (.binary (some .fp32) .real .mul (shape a) (shape b)))=.right) :
    eval I S (.binary (some .fp32) .real .add c (.binary (some .fp32) .real .mul a b))=
      I.fmaRN (eval I S a) (eval I S b) (eval I S c) := by simp [eval,shape,h]

/-- The memory injection retains every discrete metadata cell unchanged. -/
def inject (r : RegionName) (o : Nat) : (d : TileDType) → Value Word d → Value Term d
  | .real,w => .leaf r o w
  | .fp32,w => .leaf r o w
  | .fp16,w => .leaf r o w
  | .bf16,w => .leaf r o w
  | .f8e4,w => .leaf r o w
  | .f8e5,w => .leaf r o w
  | .nat,n => n
  | .int,n => n
  | .bool,n => n
  | .ptr,n => n
  | .blockPtr,n => n

def entry (m : Grid.Memory Word) : Grid.Memory Term := fun r o =>
  match m r o with | .mk d v => .mk d (inject r o d v)

def observe (I : Instructions) (S : Schedule) (m : Grid.Memory Term) (r : RegionName) (o : Nat) :=
  eval I S ((m r o).read .fp32)

theorem complete (I : Instructions) (R : Reductions) (S : Schedule)
    (p : Parameters) (work : List Grid.PID) (memory : Grid.Memory Word)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ final, Grid.run algebra (reduction R) p work (entry memory)=some final ∧
      (∀ e∈Grid.events p work, observe I S final p.output (Grid.offset p e.1 e.2)=
        eval I S (Grid.values algebra (reduction R) p (entry memory) e)) ∧
      (∀ r o, (r≠p.output ∨ ∀ e∈Grid.events p work,o≠Grid.offset p e.1 e.2) →
        final r o=entry memory r o) := by
  obtain ⟨final,run,_,readback,frame⟩ := Grid.whole algebra (reduction R) p work (entry memory) separate unique
  refine ⟨final,run,?_,frame⟩
  intro e live
  simp only [observe,readback e live,Cell.read_same]

structure InstructionAgreement (a b : Instructions) : Prop where
  addRN : ∀ x y, a.addRN x y=b.addRN x y
  mulRN : ∀ x y, a.mulRN x y=b.mulRN x y
  fmaRN : ∀ x y z, a.fmaRN x y z=b.fmaRN x y z

theorem instruction_equality (a b : Instructions) (h : InstructionAgreement a b) : a=b := by
  cases a; cases b
  have h1 := funext fun x => funext (h.addRN x)
  have h2 := funext fun x => funext (h.mulRN x)
  have h3 := funext fun x => funext fun y => funext (h.fmaRN x y)
  simp_all

theorem eval_agreement (a b : Instructions) (h : InstructionAgreement a b) (s : Schedule) (e : Term) :
    eval a s e=eval b s e := by rw [instruction_equality a b h]

#axiomsClean instruction_equality
#axiomsClean eval_agreement
#axiomsClean Plan.lane_coverage
#axiomsClean left_fused
#axiomsClean right_fused
#axiomsClean complete
#specNonCircular eval avoiding [FP.Structural.exec,Grid.run]
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Native
