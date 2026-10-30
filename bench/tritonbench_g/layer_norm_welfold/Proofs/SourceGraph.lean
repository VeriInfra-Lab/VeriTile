import bench.tritonbench_g.layer_norm_welfold.Proofs.Host
import bench.tritonbench_g.layer_norm_welfold.Proofs.ScalarTransport
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Native
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000

/-- Only scalar source operations are bound. No reduction, row or output field. -/
structure SourceBindings {α : Type} (M : Algebra α) (p : Primitives α) : Prop where
  zero : p.zero = LayerNormWelfold.zero M
  one : p.one = M.fromNat (some .fp32) 1
  add : ∀ x y, p.add x y = M.binary (some .fp32) .real .add x y
  sub : ∀ x y, p.sub x y = M.binary (some .fp32) .real .sub x y
  mul : ∀ x y, p.mul x y = M.binary (some .fp32) .real .mul x y
  div : ∀ x y, p.div x y = M.binary (some .fp32) .real .div x y
  cast32 : ∀ x, p.unary 0 x = M.cast (some .fp32) .real .real x
  bf16 : ∀ x, p.unary 1 x = M.cast none .real .bf16 x
  rsqrt : ∀ x, p.unary 2 x = M.unary (some .fp32) .rsqrt x

/-- Atom kinds 0/1 are D/epsilon; kinds 2/3/4 are original X/W/bias cells.
There is no atom representing a desired reduction or output. -/
def inputs {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row : Nat) (n : Nat) : α :=
  let pair := Nat.unpair n
  match pair.1 with
  | 0 => M.fromNat (some .fp32) D
  | 1 => M.literal none .real (1e-05)
  | 2 => (s.mem X (pair.2+D*row)).read .real
  | 3 => (s.mem W pair.2).read .real
  | _ => (s.mem Bias pair.2).read .real

def loadExpr (kind col : Nat) : Expr := .unary 0 (.unary 0 (.atom (Nat.pair kind col)))
def dimensionExpr (D : Nat) : Expr := if D=1 then .one else .atom (Nat.pair 0 0)
def laneExpr (D B k : Nat) (i : Fin B) : Expr :=
  .unary 0 (.unary 0 (if k*B+i.val<D then .atom (Nat.pair 2 (k*B+i.val)) else .zero))
def advanceExpr (phase : Bool) (D B k : Nat) (mean : Expr) (previous : Fin B→Expr) : Fin B→Expr := fun i =>
  let x := laneExpr D B k i
  let delta := Expr.sub x mean
  .add (previous i) (if phase then if k*B+i.val<D then .mul delta delta else .zero else x)
def recurrenceExpr (phase : Bool) (D B : Nat) (mean : Expr) : Nat→Nat→(Fin B→Expr)→(Fin B→Expr)
  | _,0,previous => previous
  | k,n+1,previous => recurrenceExpr phase D B mean (k+1) n (advanceExpr phase D B k mean previous)
def treeExpr {B : Nat} (values : Fin B→Expr) : ReductionTree B→Expr
  | .leaf i => values i
  | .node a b => .add (treeExpr values a) (treeExpr values b)
def totalExpr (phase : Bool) (D B : Nat) (mean : Expr) (tree : ReductionTree B) : Expr :=
  treeExpr (recurrenceExpr phase D B mean 0 (blockCount D B) (fun _ => .zero)) tree
def meanExpr (D B : Nat) (tree : ReductionTree B) : Expr :=
  .div (totalExpr Bool.false D B .zero tree) (dimensionExpr D)
def inverseExpr (D B : Nat) (tree : ReductionTree B) : Expr :=
  .unary 2 (.add (.div (totalExpr Bool.true D B (meanExpr D B tree) tree) (dimensionExpr D))
    (.atom (Nat.pair 1 0)))
def outputExpr (D B col : Nat) (tree : ReductionTree B) : Expr :=
  .unary 1 (.unary 0 (.add (.mul (.mul (.sub (loadExpr 2 col) (meanExpr D B tree))
    (inverseExpr D B tree)) (loadExpr 3 col)) (loadExpr 4 col)))

theorem dimension_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D row : Nat) :
    eval p (inputs M s X W Bias D row) (dimensionExpr D)=M.fromNat (some .fp32) D := by
  by_cases hd : D=1 <;> simp [dimensionExpr,hd,eval,inputs,h.one]

theorem lane_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row k : Nat) (i : Fin B) :
    eval p (inputs M s X W Bias D row) (laneExpr D B k i)=loaded M (atRow s row) X D B k i := by
  by_cases hc : k*B+i.val<D <;> simp [laneExpr,eval,h.cast32,h.zero,inputs,loaded,atRow,hc]

theorem recurrence_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row : Nat)
    (phase : Bool) (mean : Expr) (k n : Nat) (previous : Fin B→Expr) :
    (fun i => eval p (inputs M s X W Bias D row) (recurrenceExpr phase D B mean k n previous i))=
      recurrence M phase (atRow s row) X D B (eval p (inputs M s X W Bias D row) mean) k n
        (fun i => eval p (inputs M s X W Bias D row) (previous i)) := by
  induction n generalizing k previous with
  | zero => rfl
  | succ n ih =>
    simp only [recurrenceExpr,recurrence,ih]
    congr 1
    funext i
    cases phase <;> by_cases hc : k*B+i.val<D <;>
      simp [advanceExpr,advance,eval,lane_eval M p h,h.add,h.sub,h.mul,h.zero,hc]

theorem tree_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (input : Nat→α) {B : Nat} (values : Fin B→Expr) (tree : ReductionTree B) :
    eval p input (treeExpr values tree)=treeValue M (fun i => eval p input (values i)) tree := by
  induction tree with
  | leaf i => rfl
  | node a b ha hb => simp [treeExpr,eval,treeValue,h.add,ha,hb]

theorem total_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row : Nat)
    (phase : Bool) (mean : Expr) (tree : ReductionTree B) :
    eval p (inputs M s X W Bias D row) (totalExpr phase D B mean tree)=
      treeValue M (allLanes M phase (atRow s row) X D B (eval p (inputs M s X W Bias D row) mean)) tree := by
  simp only [totalExpr,tree_eval M p h,recurrence_eval M p h,eval,h.zero,allLanes]

theorem mean_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row : Nat) (tree : ReductionTree B) :
    eval p (inputs M s X W Bias D row) (meanExpr D B tree)=meanValue M (atRow s row) X D B tree := by
  simp [meanExpr,eval,total_eval M p h,dimension_eval M p h,h.div,h.zero,meanValue]

theorem inverse_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row : Nat) (tree : ReductionTree B) :
    eval p (inputs M s X W Bias D row) (inverseExpr D B tree)=inverseValue M (atRow s row) X D B tree := by
  simp [inverseExpr,eval,total_eval M p h,dimension_eval M p h,mean_eval M p h,
    h.div,h.add,h.rsqrt,inputs,inverseValue]

theorem output_eval {α : Type} [Inhabited α] (M : Algebra α) (p : Primitives α)
    (h : SourceBindings M p) (s : State α) (X W Bias : RegionName) (D B row col : Nat) (tree : ReductionTree B) :
    eval p (inputs M s X W Bias D row) (outputExpr D B col tree)=rowOutput M s X W Bias D B row col tree := by
  simp [outputExpr,eval,mean_eval M p h,inverse_eval M p h,loadExpr,inputs,h.bf16,h.cast32,
    h.add,h.mul,h.sub,rowOutput,normalizeValue]

#axiomsClean recurrence_eval
#axiomsClean mean_eval
#axiomsClean inverse_eval
#axiomsClean output_eval
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Native
