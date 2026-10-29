import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerOperands

/-! Source-site recipes for the selected FP32 compiler lowering. These keep
the actual primitive operand orders and optional branches. Reduction recipes
consume the selected compiler artifact's ordered trees, with an owner tag at
every internal node and the leaf's own mean owner inside variance. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Compiler
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000

def read (input : Input) : Term := .at (.load input) (.cast32 (.input input))
def maskedRead (input : Input) (N col : Nat) : Term :=
  .at (.load input) (.cast32 (if col<N then .input input else .literal 0))
def random (second : Bool) (col : Nat) : Term := .at (.random second col) (.random second col)

def stream (f : Flags) (N : Nat) (second : Bool) (col : Nat) : Term :=
  .at (.stream second col) <|
    let raw := maskedRead (.x second col) N col
    let scaled := if f.rowscale then Term.mul raw (read (.rowscale second)) else raw
    if f.dropout then .selectGT (random second col) (.input .dropoutP)
      (.div scaled (.sub (.literal 1) (.input .dropoutP))) (.literal 0) else scaled

def combined (f : Flags) (N col : Nat) : Term :=
  .at (.combined col) <|
    let main := stream f N Bool.false col
    let two := if f.x1 then Term.add main (stream f N Bool.true col) else main
    if f.residual then
      if f.rowscale && !f.dropout && !f.x1 then
        .fma (maskedRead (.x Bool.false col) N col) (read (.rowscale Bool.false))
          (maskedRead (.residual col) N col)
      else .add two (maskedRead (.residual col) N col)
    else two

def sum (owner : Nat) (path : List Bool) : Whole.SumTree block → (Fin block → Term) → Term
  | .leaf j,x => .at (.meanTree owner path) (x j)
  | .add l r,x => .at (.meanTree owner path)
      (.add (sum owner (path++[Bool.false]) l x) (sum owner (path++[Bool.true]) r x))

def squares (owner : Nat) (path : List Bool) : Whole.SquareTree block → (Fin block → Term) → Term
  | .square j,x => .at (.varianceTree owner path) (.mul (x j) (x j))
  | .add l r,x => .at (.varianceTree owner path)
      (.add (squares owner (path++[Bool.false]) l x) (squares owner (path++[Bool.true]) r x))
  | .fmaSquare j r,x => .at (.varianceTree owner path)
      (.fma (x j) (x j) (squares owner (path++[Bool.true]) r x))

def mean (red : Nat → Whole.Reductions block) (f : Flags) (N owner : Nat) : Term :=
  .at (.mean owner) <| if N=1 then combined f N 0
    else .div (sum owner [] (red owner).mean (fun j => combined f N j.val)) (.literal N)
def centered (red : Nat → Whole.Reductions block) (f : Flags) (N owner : Nat) : Term :=
  .at (.centered owner) <| if f.rms then combined f N owner
    else .sub (combined f N owner) (mean red f N owner)
def variance (red : Nat → Whole.Reductions block) (f : Flags) (N owner : Nat) : Term :=
  .at (.variance owner) <| .div (squares owner [] (red owner).variance
    (fun j => if j.val<N then centered red f N j.val else .literal 0)) (.literal N)
def rstd (red : Nat → Whole.Reductions block) (f : Flags) (N owner : Nat) : Term :=
  .at (.rstd owner) <| .div (.literal 1) (.sqrt (if N=1 then
    .fma (centered red f N 0) (centered red f N 0) (.input .eps)
    else .add (.input .eps) (variance red f N owner)))
def normalized (red : Nat → Whole.Reductions block) (f : Flags) (N owner : Nat) : Term :=
  .at (.normalized owner) <| if f.rms then
    .mul (rstd red f N owner) (centered red f N owner)
    else .mul (centered red f N owner) (rstd red f N owner)
def affine (plan : AffinePlan) (bias : Bool) (x w b : Term) : Term :=
  .cast32 (if bias then match plan with
    | .separate => .add (.mul x w) b
    | .fused => .fma x w b
  else .mul x w)
def y (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags)
    (N : Nat) (second : Bool) (col : Nat) : Term :=
  .at (.storeY second col) <| .at (.affine second col) <|
    affine plan (if second then f.b1 else f.bias) (normalized red f N col)
      (read (.weight second col)) (read (.bias second col))
def residual (f : Flags) (N col : Nat) : Term :=
  .at (.storeResidual col) (.cast32 (combined f N col))
def storedMean (red : Nat → Whole.Reductions block) (f : Flags) (N : Nat) : Term :=
  .at .storeMean (.cast32 (mean red f N 0))
def storedRstd (red : Nat → Whole.Reductions block) (f : Flags) (N : Nat) : Term :=
  .at .storeRstd (.cast32 (rstd red f N 0))

theorem stream_instantiate (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α)
    (row col : Nat) (second : Bool) :
    (stream f N second col).instantiate x M row eps p=
      NativeParameters.stream f x M N p second row col := by
  cases second <;> cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [stream,maskedRead,read,random,Term.instantiate,Input.value,
    NativeParameters.stream,NativeRow.load,hs,hd]
  all_goals split_ifs <;> simp [Term.instantiate,Input.value]

theorem combined_instantiate (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α)
    (row col : Nat) :
    (combined f N col).instantiate x M row eps p=NativeParameters.combined f x M N p row col := by
  simp only [combined,NativeParameters.combined,Term.instantiate]
  split_ifs <;> simp [stream_instantiate,maskedRead,read,Term.instantiate,Input.value,NativeRow.load]
  all_goals split_ifs <;> simp [Term.instantiate,Input.value]

theorem sum_instantiate (tree : Whole.SumTree block) (values : Fin block → Term)
    (x : Inputs α) (M row : Nat) (eps p : α) (owner : Nat) (path : List Bool) :
    (sum owner path tree values).instantiate x M row eps p=
      sumExpr tree (fun j => (values j).instantiate x M row eps p) := by
  induction tree generalizing path with
  | leaf j => rfl
  | add l r hl hr => simp only [sum,Term.instantiate,sumExpr,hl,hr]

theorem squares_instantiate (tree : Whole.SquareTree block) (values : Fin block → Term)
    (x : Inputs α) (M row : Nat) (eps p : α) (owner : Nat) (path : List Bool) :
    (squares owner path tree values).instantiate x M row eps p=
      squareExpr tree (fun j => (values j).instantiate x M row eps p) := by
  induction tree generalizing path with
  | square j => rfl
  | add l r hl hr => simp only [squares,Term.instantiate,squareExpr,hl,hr]
  | fmaSquare j r hr => simp only [squares,Term.instantiate,squareExpr,hr]

theorem mean_instantiate (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row owner : Nat) :
    (mean red f N owner).instantiate x M row eps p=NativeParameters.mean red f x M N p row owner := by
  simp only [mean,NativeParameters.mean,Term.instantiate]
  split_ifs <;> simp only [Term.instantiate,sum_instantiate,combined_instantiate]

theorem centered_instantiate (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row owner : Nat) :
    (centered red f N owner).instantiate x M row eps p=NativeParameters.centered red f x M N p row owner := by
  simp only [centered,NativeParameters.centered,Term.instantiate]
  split_ifs <;> simp only [Term.instantiate,mean_instantiate,combined_instantiate]

theorem variance_instantiate (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row owner : Nat) :
    (variance red f N owner).instantiate x M row eps p=NativeParameters.variance red f x M N p row owner := by
  simp only [variance,NativeParameters.variance,Term.instantiate,squares_instantiate]
  congr 2
  funext j
  split_ifs <;> simp only [centered_instantiate,Term.instantiate]

theorem rstd_instantiate (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row owner : Nat) :
    (rstd red f N owner).instantiate x M row eps p=NativeParameters.rstd red f x M N eps p row owner := by
  simp only [rstd,NativeParameters.rstd,Term.instantiate]
  split_ifs <;> simp only [Term.instantiate,Input.value,centered_instantiate,variance_instantiate]

theorem normalized_instantiate (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row owner : Nat) :
    (normalized red f N owner).instantiate x M row eps p=NativeParameters.normalized red f x M N eps p row owner := by
  simp only [normalized,NativeParameters.normalized,Term.instantiate]
  split_ifs <;> simp only [Term.instantiate,centered_instantiate,rstd_instantiate]

theorem affine_instantiate (plan : AffinePlan) (bias : Bool) (a w b : Term)
    (x : Inputs α) (M row : Nat) (eps p : α) :
    (affine plan bias a w b).instantiate x M row eps p=
      NativeRow.affine plan bias (a.instantiate x M row eps p)
        (w.instantiate x M row eps p) (b.instantiate x M row eps p) := by
  cases plan <;> cases bias <;> rfl

theorem y_instantiate (red : Nat → Whole.Reductions block) (plan : AffinePlan)
    (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α) (row col : Nat) :
    (y red plan f N Bool.false col).instantiate x M row eps p=
      NativeParameters.y red plan f x M N eps p row col := by
  simp only [y,Term.instantiate,affine_instantiate,normalized_instantiate,read,Input.value,NativeParameters.y]
  rfl

theorem y1_instantiate (red : Nat → Whole.Reductions block) (plan : AffinePlan)
    (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α) (row col : Nat) :
    (y red plan f N Bool.true col).instantiate x M row eps p=
      NativeParameters.y1 red plan f x M N eps p row col := by
  simp only [y,Term.instantiate,affine_instantiate,normalized_instantiate,read,Input.value,NativeParameters.y1]
  rfl

#axiomsClean stream_instantiate
#axiomsClean combined_instantiate
#axiomsClean sum_instantiate
#axiomsClean squares_instantiate
#axiomsClean mean_instantiate
#axiomsClean centered_instantiate
#axiomsClean variance_instantiate
#axiomsClean rstd_instantiate
#axiomsClean normalized_instantiate
#axiomsClean affine_instantiate
#axiomsClean y_instantiate
#axiomsClean y1_instantiate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Compiler
