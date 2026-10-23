import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.FP
import bench.tritonbench_g.layer_norm_fwd.Proofs.Formula

/-! Ordered scalar/reduction expression model. Each column retains its
own reduction trees, including the different mean trees used to center the
leaves of a later variance reduction; butterfly lanes are not conflated. Reduction schedules are
explicit trees and affine contractions are explicit constructors. The optional
exact-Real ports below are mathematical hypotheses used only by the historical
conditional Real theorem. They are generally false for rounded native
arithmetic and are not pending native-conformance obligations. The opaque
evaluator itself uses no such hypotheses. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeRow
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 5000

structure Ops (α : Type) extends Whole.Numerics α where
  decode : α → ℝ
  cast32 : α → α

inductive Expr (α : Type) where
  | input (value : α)
  | literal (value : ℝ)
  | random (seed : Int) (col : Nat)
  | cast32 (x : Expr α)
  | add (x y : Expr α) | sub (x y : Expr α) | mul (x y : Expr α) | div (x y : Expr α)
  | fma (x y z : Expr α)
  | sqrt (x : Expr α)
  | selectGT (x y yes no : Expr α)

def Expr.eval (a : Ops α) : Expr α → α
  | .input x => x | .literal x => a.literal x
  | .random seed col => a.randomValue (Random.rand (Formula.seedU32 seed) (col : Int) .bits32 7)
  | .cast32 x => a.cast32 (x.eval a)
  | .add x y => a.add (x.eval a) (y.eval a)
  | .sub x y => a.sub (x.eval a) (y.eval a)
  | .mul x y => a.mul (x.eval a) (y.eval a)
  | .div x y => a.div (x.eval a) (y.eval a)
  | .fma x y z => a.fma (x.eval a) (y.eval a) (z.eval a)
  | .sqrt x => a.sqrt (x.eval a)
  | .selectGT x y yes no => if a.greater (x.eval a) (y.eval a) then yes.eval a else no.eval a

def Expr.real (a : Ops α) : Expr α → ℝ
  | .input x => a.decode x | .literal x => x
  | .random seed col => Random.randReal (Formula.seedU32 seed) (col : Int) .bits32 7
  | .cast32 x => x.real a
  | .add x y => x.real a+y.real a | .sub x y => x.real a-y.real a
  | .mul x y => x.real a*y.real a | .div x y => x.real a/y.real a
  | .fma x y z => x.real a*y.real a+z.real a
  | .sqrt x => Real.sqrt (x.real a)
  | .selectGT x y yes no => if decide (x.real a>y.real a) then yes.real a else no.real a

/-- Hypothetical exactness at each node. This is not a valid general native
FP32 accuracy contract: even finite positive sqrt(2) has no exact FP32 result.
All children of select are included. Opaque execution must not assume this. -/
def Expr.Ports (a : Ops α) : Expr α → Prop
  | .input _ => True
  | .literal x => a.decode (a.literal x)=x
  | .random seed col => a.decode (a.randomValue (Random.rand (Formula.seedU32 seed) (col : Int) .bits32 7)) =
      Random.randReal (Formula.seedU32 seed) (col : Int) .bits32 7
  | .cast32 x => x.Ports a ∧ a.decode (a.cast32 (x.eval a))=a.decode (x.eval a)
  | .add x y => x.Ports a ∧ y.Ports a ∧ a.decode (a.add (x.eval a) (y.eval a))=a.decode (x.eval a)+a.decode (y.eval a)
  | .sub x y => x.Ports a ∧ y.Ports a ∧ a.decode (a.sub (x.eval a) (y.eval a))=a.decode (x.eval a)-a.decode (y.eval a)
  | .mul x y => x.Ports a ∧ y.Ports a ∧ a.decode (a.mul (x.eval a) (y.eval a))=a.decode (x.eval a)*a.decode (y.eval a)
  | .div x y => x.Ports a ∧ y.Ports a ∧ a.decode (a.div (x.eval a) (y.eval a))=a.decode (x.eval a)/a.decode (y.eval a)
  | .fma x y z => x.Ports a ∧ y.Ports a ∧ z.Ports a ∧
      a.decode (a.fma (x.eval a) (y.eval a) (z.eval a))=a.decode (x.eval a)*a.decode (y.eval a)+a.decode (z.eval a)
  | .sqrt x => x.Ports a ∧ a.decode (a.sqrt (x.eval a))=Real.sqrt (a.decode (x.eval a))
  | .selectGT x y yes no => x.Ports a ∧ y.Ports a ∧ yes.Ports a ∧ no.Ports a ∧
      a.greater (x.eval a) (y.eval a)=decide (a.decode (x.eval a)>a.decode (y.eval a))

theorem Expr.ports_sound (a : Ops α) (e : Expr α) (h : e.Ports a) :
    a.decode (e.eval a)=e.real a := by
  induction e with
  | input x => rfl
  | literal x => exact h
  | random s c => exact h
  | cast32 x hx => exact h.2.trans (hx h.1)
  | add x y hx hy => exact h.2.2.trans (congrArg₂ (·+·) (hx h.1) (hy h.2.1))
  | sub x y hx hy => exact h.2.2.trans (congrArg₂ (·-·) (hx h.1) (hy h.2.1))
  | mul x y hx hy => exact h.2.2.trans (congrArg₂ (·*·) (hx h.1) (hy h.2.1))
  | div x y hx hy => exact h.2.2.trans (congrArg₂ (·/·) (hx h.1) (hy h.2.1))
  | fma x y z hx hy hz =>
    exact h.2.2.2.trans (congrArg₂ (·+·) (congrArg₂ (·*·) (hx h.1) (hy h.2.1)) (hz h.2.2.1))
  | sqrt x hx => exact h.2.trans (congrArg Real.sqrt (hx h.1))
  | selectGT x y yes no hx hy ht hf =>
    simp only [Expr.eval,Expr.real,h.2.2.2.2,hx h.1,hy h.2.1]
    split <;> first | exact ht h.2.2.1 | exact hf h.2.2.2.1

structure Inputs (α : Type) where
  x : Nat → Nat → α
  x1 : Nat → Nat → α
  residual : Nat → Nat → α
  weight : Nat → α
  bias : Nat → α
  weight1 : Nat → α
  bias1 : Nat → α
  rowscale : Nat → α
  seeds : Nat → Int

def Inputs.decode (a : Ops α) (x : Inputs α) : VeriTile.Bench.TritonBenchG.LayerNormFwd.Inputs where
  x := fun i j => a.decode (x.x i j)
  x1 := fun i j => a.decode (x.x1 i j)
  residual := fun i j => a.decode (x.residual i j)
  weight := fun j => a.decode (x.weight j)
  bias := fun j => a.decode (x.bias j)
  weight1 := fun j => a.decode (x.weight1 j)
  bias1 := fun j => a.decode (x.bias1 j)
  rowscale := fun i => a.decode (x.rowscale i)
  seeds := x.seeds

def load (x : Nat → Nat → α) (N row col : Nat) : Expr α :=
  .cast32 (if col<N then .input (x row col) else .literal 0)

def stream (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ)
    (second : Bool) (row col : Nat) : Expr α :=
  let raw := load (if second then x.x1 else x.x) N row col
  let index := if second then M+row else row
  let scaled := if f.rowscale then Expr.mul raw (.cast32 (.input (x.rowscale index))) else raw
  if f.dropout then .selectGT (.random (x.seeds index) col) (.literal p)
    (.div scaled (.sub (.literal 1) (.literal p))) (.literal 0) else scaled

def combined (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ) (row col : Nat) : Expr α :=
  let main := stream f x M N p Bool.false row col
  let two := if f.x1 then Expr.add main (stream f x M N p Bool.true row col) else main
  if f.residual then
    if f.rowscale && !f.dropout && !f.x1 then
      .fma (load x.x N row col) (.cast32 (.input (x.rowscale row))) (load x.residual N row col)
    else .add two (load x.residual N row col)
  else two

def sumExpr : Whole.SumTree block → (Fin block → Expr α) → Expr α
  | .leaf j,x => x j
  | .add l r,x => .add (sumExpr l x) (sumExpr r x)
def squareExpr : Whole.SquareTree block → (Fin block → Expr α) → Expr α
  | .square j,x => .mul (x j) (x j)
  | .add l r,x => .add (squareExpr l x) (squareExpr r x)
  | .fmaSquare j r,x => .fma (x j) (x j) (squareExpr r x)

theorem sumExpr_real (a : Ops α) (t : Whole.SumTree block) (x : Fin block → Expr α) :
    (sumExpr t x).real a = t.eval (·+·) (fun j => (x j).real a) := by
  induction t with
  | leaf j => rfl
  | add l r hl hr => exact congrArg₂ (·+·) hl hr

theorem squareExpr_real (a : Ops α) (t : Whole.SquareTree block) (x : Fin block → Expr α) :
    (squareExpr t x).real a = t.eval (·*·) (·+·) (fun a b c => a*b+c) (fun j => (x j).real a) := by
  induction t with
  | square j => rfl
  | add l r hl hr => exact congrArg₂ (·+·) hl hr
  | fmaSquare j r hr => exact congrArg (fun z => (x j).real a*(x j).real a+z) hr

def mean (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ) (row col : Nat) : Expr α :=
  if N=1 then combined f x M N p row 0
  else .div (sumExpr (red col).mean (fun j => combined f x M N p row j.val)) (.literal N)
def centered (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ) (row col : Nat) : Expr α :=
  if f.rms then combined f x M N p row col
  else .sub (combined f x M N p row col) (mean red f x M N p row col)
def variance (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ) (row col : Nat) : Expr α :=
  .div (squareExpr (red col).variance (fun j => if j.val<N then centered red f x M N p row j.val else .literal 0)) (.literal N)
def rstd (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (eps p : ℝ) (row col : Nat) : Expr α :=
  .div (.literal 1) (.sqrt (if N=1 then
    .fma (centered red f x M N p row 0) (centered red f x M N p row 0) (.literal eps)
    else .add (.literal eps) (variance red f x M N p row col)))
def normalized (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (eps p : ℝ) (row col : Nat) : Expr α :=
  if f.rms then .mul (rstd red f x M N eps p row col) (centered red f x M N p row col)
  else .mul (centered red f x M N p row col) (rstd red f x M N eps p row col)

inductive AffinePlan where | separate | fused
  deriving DecidableEq

def affine (plan : AffinePlan) (bias : Bool) (x w b : Expr α) : Expr α :=
  .cast32 (if bias then match plan with
    | .separate => .add (.mul x w) b
    | .fused => .fma x w b
  else .mul x w)

def y (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) : Expr α :=
  affine plan f.bias (normalized red f x M N eps p row col)
    (.cast32 (.input (x.weight col))) (.cast32 (.input (x.bias col)))
def y1 (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) : Expr α :=
  affine plan f.b1 (normalized red f x M N eps p row col)
    (.cast32 (.input (x.weight1 col))) (.cast32 (.input (x.bias1 col)))

theorem stream_real (a : Ops α) (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ)
    (second : Bool) (row col : Nat) :
    (stream f x M N p second row col).real a =
      Formula.stream f (Formula.paddedInputs (x.decode a) N) M p second row col := by
  cases second <;> cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [stream,load,Expr.real,Formula.stream,Formula.dropout,Formula.keep,
    Formula.paddedInputs,Inputs.decode,hs,hd]
  all_goals split_ifs <;> simp_all [Expr.real]

theorem combined_real (a : Ops α) (f : Flags) (x : Inputs α) (M N : Nat) (p : ℝ)
    (row col : Nat) :
    (combined f x M N p row col).real a =
      if col<N then Formula.combined f (x.decode a) M p row col else 0 := by
  have hpad := Formula.combined_padded f (x.decode a) M N p row col
  rw [←hpad]
  unfold combined Formula.combined
  cases hx : f.x1 <;> cases hr : f.residual <;> cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [Expr.real,stream,load,Formula.stream,Formula.dropout,Formula.keep,Formula.paddedInputs,Inputs.decode,hx,hr,hs,hd]
  all_goals split_ifs <;> simp_all [Expr.real]

theorem mean_real (a : Ops α) (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (p : ℝ) (row col : Nat) (hN : N≤block) :
    (mean red f x M N p row col).real a = Formula.mean f (x.decode a) M N p row := by
  by_cases h1 : N=1
  · subst N; simp [mean,combined_real,Formula.mean]
  · simp only [mean,if_neg h1,Expr.real,sumExpr_real,combined_real]
    rw [Whole.mean_padded (red col) _ hN]
    rfl

theorem centered_real (a : Ops α) (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (p : ℝ) (row col : Nat) (hN : N≤block) (hc : col<N) :
    (centered red f x M N p row col).real a = Formula.centered f (x.decode a) M N p row col := by
  cases hr : f.rms <;> simp [centered,Expr.real,Formula.centered,hr,combined_real,mean_real,hN,hc]

theorem variance_real (a : Ops α) (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (p : ℝ) (row col : Nat) (hN : N≤block) :
    (variance red f x M N p row col).real a = Formula.variance f (x.decode a) M N p row := by
  simp only [variance,Expr.real,squareExpr_real]
  have hvalues : (fun j : Fin block => (if j.val<N then centered red f x M N p row j.val else Expr.literal 0).real a) =
      (fun j : Fin block => if j.val<N then Formula.centered f (x.decode a) M N p row j.val else 0) := by
    funext j
    split_ifs with hj
    · exact centered_real a red f x M N p row j.val hN hj
    · rfl
  rw [hvalues,Whole.variance_padded (red col) _ hN]
  rfl

theorem rstd_real (a : Ops α) (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (hN : N≤block) :
    (rstd red f x M N eps p row col).real a = Formula.rstd f (x.decode a) M N eps p row := by
  by_cases h1 : N=1
  · subst N
    simp [rstd,Expr.real,centered_real,hN,Formula.rstd,Formula.variance,pow_two]
  · simp [rstd,h1,Expr.real,variance_real,hN,Formula.rstd,add_comm]

theorem y_real (a : Ops α) (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (hN : N≤block) (hc : col<N) :
    (y red plan f x M N eps p row col).real a = Formula.y f (x.decode a) M N eps p row col := by
  cases plan <;> cases hb : f.bias <;> cases hr : f.rms
  all_goals simp [y,affine,normalized,Expr.real,centered_real,rstd_real,hN,hc,hb,hr,mul_comm,
    Formula.y,Formula.normalized,Inputs.decode]

theorem y1_real (a : Ops α) (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (hN : N≤block) (hc : col<N) :
    (y1 red plan f x M N eps p row col).real a = Formula.y1 f (x.decode a) M N eps p row col := by
  cases plan <;> cases hb : f.b1 <;> cases hr : f.rms
  all_goals simp [y1,affine,normalized,Expr.real,centered_real,rstd_real,hN,hc,hb,hr,mul_comm,
    Formula.y1,Formula.normalized,Inputs.decode]

/-- Whole active output from operand-local ports; all reduction nodes and
native affine FMA/separate choices are included in the expression. -/
theorem y_ports (a : Ops α) (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (hN : N≤block) (hc : col<N)
    (h : (y red plan f x M N eps p row col).Ports a) :
    a.decode ((y red plan f x M N eps p row col).eval a) = Formula.y f (x.decode a) M N eps p row col :=
  (Expr.ports_sound a _ h).trans (y_real a red plan f x M N eps p row col hN hc)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeRow
