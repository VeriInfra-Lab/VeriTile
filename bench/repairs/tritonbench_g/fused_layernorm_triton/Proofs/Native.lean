import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Kernel
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
namespace Compiler
export VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
  (Term Shape Instructions Fusion shape)
end Compiler
abbrev Word := BitVec 32
abbrev Term := Compiler.Term
noncomputable section

/-- Individual native arithmetic/conversion ports plus IEEE-style equality.
No finite-input restriction or relationship to exact Real arithmetic is assumed. -/
structure Instructions extends Compiler.Instructions where
  equal : Word → Word → Bool

/-- Native addition and multiplication may each select their operand order.
The schedule sees operation shape and physical leaf identity, never raw input
words. In particular NaN payload order is not silently quotiented away. -/
structure Schedule where
  fusion : Compiler.Shape → Compiler.Fusion
  swapMultiply : Compiler.Shape → Bool
  swapAdd : Compiler.Shape → Bool

/-- Explicit scalar native lowering. Adjacent mul/add contraction emits a genuine
ternary instruction. Casts and physical leaves remain expression boundaries. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => I.literal p d x
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (Compiler.shape a) (Compiler.shape b)), p, d, a, b with
    | .left, some .fp32, .real, .binary (some .fp32) .real .mul x y, c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .right, some .fp32, .real, c, .binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .leftSwap, some .fp32, .real, .binary (some .fp32) .real .mul x y, c =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | .rightSwap, some .fp32, .real, c, .binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | _, _, _, a, b =>
      if S.swapAdd (.binary p d .add (Compiler.shape a) (Compiler.shape b)) then
        I.binary p d .add (eval I S b) (eval I S a)
      else I.binary p d .add (eval I S a) (eval I S b)
  | .binary p d .mul a b =>
    if S.swapMultiply (.binary p d .mul (Compiler.shape a) (Compiler.shape b)) then
      I.binary p d .mul (eval I S b) (eval I S a)
    else I.binary p d .mul (eval I S a) (eval I S b)
  | .binary p d op a b => I.binary p d op (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast p src dst a => I.cast p src dst (eval I S a)
  | .fromNat p n => I.fromNat p n
  | .fromInt p n => I.fromInt p n
  | .bits w => w
  | .load a => eval I S a

/-- The compiler profile chooses contractions and operand ordering from operation
shape only. Every adjacent FP32 mul/add site in this kernel contracts in the
observed SM80 profile; which multiplicand comes first remains explicit. -/
structure Profile where
  schedule : Schedule
  rightFused : ∀ a b c, schedule.fusion (.binary (some .fp32) .real .add c
    (.binary (some .fp32) .real .mul a b)) = .right ∨
    schedule.fusion (.binary (some .fp32) .real .add c
      (.binary (some .fp32) .real .mul a b)) = .rightSwap
  leftFused : ∀ a b c, (∀ x y, c ≠ .binary (some .fp32) .real .mul x y) →
    schedule.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) c) = .left ∨
    schedule.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) c) = .leftSwap

def observe (I : Instructions) (P : Profile) (t : Term) : Word :=
  eval I P.schedule t

def equal (I : Instructions) (P : Profile) (a b : Term) : Bool :=
  I.equal (observe I P a) (observe I P b)

/-- No whole-reduction callback is used: the explicit helper tree below expands
all its arithmetic. These unused reduction fields cannot supply any output. -/
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
  reduceMax := fun _ {_} _ _ _ => fun _ => .negInf
  reduceSum := fun p {_} _ _ _ => fun _ => .literal p .real 0

/-- Independent ordered Welford recurrence over logical input values. Its
arguments contain no memory, registers, kernel, source evaluator or output. -/
def first {α : Type} (M : Algebra α) (x : Nat → α) (D B : Nat) : Fin B → WelfordTriple α := fun i =>
  if i.val < D then
    ⟨M.cast (some .fp32) .real .real (M.cast (some .fp32) .real .real (x i.val)),
      M.literal (some .fp32) .real 0, M.literal (some .fp32) .real 1⟩
  else ⟨M.literal (some .fp32) .real 0, M.literal (some .fp32) .real 0, M.literal (some .fp32) .real 0⟩

def update {α : Type} (M : Algebra α) (value : α) (v : WelfordTriple α) : WelfordTriple α :=
  let delta := M.binary (some .fp32) .real .sub value v.mean
  let weight := M.binary (some .fp32) .real .add v.weight (M.literal (some .fp32) .real 1)
  let mean := M.binary (some .fp32) .real .add v.mean (M.binary (some .fp32) .real .div delta weight)
  ⟨mean, M.binary (some .fp32) .real .add v.m2
    (M.binary (some .fp32) .real .mul delta (M.binary (some .fp32) .real .sub value mean)), weight⟩

def advance {α : Type} (M : Algebra α) (x : Nat → α) (D B k : Nat)
    (v : Fin B → WelfordTriple α) : Fin B → WelfordTriple α := fun i =>
  if k * B + i.val < D then
    update M (M.cast (some .fp32) .real .real
      (M.cast (some .fp32) .real .real (x (k * B + i.val)))) (v i)
  else v i

def chunks {α : Type} (M : Algebra α) (x : Nat → α) (D B : Nat) :
    Nat → Nat → (Fin B → WelfordTriple α) → (Fin B → WelfordTriple α)
  | _, 0, v => v
  | start, count + 1, v => chunks M x D B (start+1) count (advance M x D B start v)

def combine {α : Type} (M : Algebra α) (eq32 : α → α → Bool)
    (a b : WelfordTriple α) : WelfordTriple α :=
  let zero := M.literal (some .fp32) .real 0
  let delta := if eq32 a.mean b.mean then zero else M.binary (some .fp32) .real .sub b.mean a.mean
  let weight := M.binary (some .fp32) .real .add a.weight b.weight
  let ratio := if eq32 weight zero then zero else M.binary (some .fp32) .real .div b.weight weight
  ⟨M.binary (some .fp32) .real .add a.mean (M.binary (some .fp32) .real .mul delta ratio),
    M.binary (some .fp32) .real .add (M.binary (some .fp32) .real .add a.m2 b.m2)
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul (M.binary (some .fp32) .real .mul delta delta) a.weight) ratio),
    weight⟩

def tree {α : Type} {B : Nat} (M : Algebra α) (eq32 : α → α → Bool)
    (v : Fin B → WelfordTriple α) : Kernels.ReductionTree B → WelfordTriple α
  | .leaf i => v i
  | .node l r => combine M eq32 (tree M eq32 v l) (tree M eq32 v r)

def statistics {α : Type} (M : Algebra α) (eq32 : α → α → Bool) (x : Nat → α)
    (D B : Nat) (t : Kernels.ReductionTree B) : WelfordTriple α :=
  tree M eq32 (chunks M x D B 1 ((D+B-1)/B-1) (first M x D B)) t

def inverse {α : Type} (M : Algebra α) (D : Nat) (v : WelfordTriple α) : α :=
  M.unary (some .fp32) .rsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div v.m2 (M.fromNat (some .fp32) D))
    (M.literal none .real (1e-5 : ℝ)))

def affine {α : Type} (M : Algebra α) (x w b avg inv : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 x)
  let w := cast32 (cast32 w)
  let b := cast32 (cast32 b)
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .sub x avg) inv) w) b))

structure Result where
  mean : Word
  rstd : Word
  output : Nat → Word

/-- Independent native target. Evaluation executes scalar instructions with the
selected FMA/operand profile, never separately rounded mul/add at fused sites. -/
def reference (I : Instructions) (P : Profile) (x w b : Nat → Term)
    (D B : Nat) (t : Kernels.ReductionTree B) : Result :=
  let v := statistics algebra (equal I P) x D B t
  let inv := inverse algebra D v
  ⟨observe I P v.mean, observe I P inv,
   fun col => observe I P (affine algebra (x col) (w col) (b col) v.mean inv)⟩

/-- Full statement of the source-to-independent graph bridge. -/
theorem first_eq {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (X : RegionName)
    (D B : Nat) (x : Nat → α) (h : ∀ j, j < D → (s.mem X (j+D*s.pids 0)).read .real = x j) :
    firstLaneValues M s X D B = first M x D B := by
  funext i
  simp only [firstLaneValues, first]
  split_ifs with hi
  · rw [h i.val hi]
  · rfl

theorem advance_eq {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (X : RegionName)
    (D B k : Nat) (x : Nat → α) (h : ∀ j, j < D → (s.mem X (j+D*s.pids 0)).read .real = x j)
    (v : Fin B → WelfordTriple α) : Whole.advance M s X D B k v = advance M x D B k v := by
  funext i
  simp only [Whole.advance, advance]
  split_ifs with hi
  · rw [h _ hi]; rfl
  · rfl

theorem chunks_eq {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (X : RegionName)
    (D B start count : Nat) (x : Nat → α)
    (h : ∀ j, j < D → (s.mem X (j+D*s.pids 0)).read .real = x j)
    (v : Fin B → WelfordTriple α) :
    laneRecurrence M s X D B start count v = chunks M x D B start count v := by
  induction count generalizing start v with
  | zero => rfl
  | succ n ih => simp only [laneRecurrence, chunks, ih, advance_eq M s X D B start x h]

theorem tree_eq {α : Type} {B : Nat} (M : Algebra α) (eq32 : α → α → Bool)
    (v : Fin B → WelfordTriple α) (t : Kernels.ReductionTree B) :
    treeValue M eq32 v (KernelProof.legacyTree t) = tree M eq32 v t := by
  induction t with
  | leaf i => rfl
  | node l r hl hr =>
    simp only [KernelProof.legacyTree,ReadableStatistics.legacyTree,treeValue,tree,hl,hr]
    norm_num only [combineTriple,combine]

theorem statistics_eq {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X : RegionName) (D B row : Nat) (t : Kernels.ReductionTree B) (x : Nat → α)
    (h : ∀ j, j < D → (s.mem X (j+D*row)).read .real = x j) :
    rowValue M eq32 (atRow s row) X D B (KernelProof.legacyTree t) = statistics M eq32 x D B t := by
  unfold rowValue allLaneValues statistics
  rw [chunks_eq M (atRow s row) X D B 1 _ x h, first_eq M (atRow s row) X D B x h, tree_eq]
  rfl

#specNonCircular reference avoiding [FP.Structural.exec, Whole.rowValue, Whole.rowOutput,
  FPStages.treeValue, Whole.laneRecurrence, Whole.runRows, Contract.source]
#axiomsClean statistics_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Native
