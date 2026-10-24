import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LocatedGraph

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

inductive RawOrigin (σ : Type) where
  | constant (site : Occurrence σ)
  | load (site : Occurrence σ) (state : σ) (tensor : Tensor) (index : Nat)

structure ReductionChoice {σ : Type} (platform : Selection σ) (kind : ReductionKind)
    (identity : Word) (width : Nat) where
  selected : SelectedTree platform kind identity width
  locations : TreeLocations selected.site

def ReductionChoice.graph {σ : Type} {platform : Selection σ} {kind identity width}
    (choice : ReductionChoice platform kind identity width) (op : ScalarBinary) : LocatedGraph σ width :=
  choice.selected.tree.locate op choice.locations.siteAt []

/-- Plan annotations are dependent on the source expression. Every reduction
occurrence has its own actually selected legal tree and exact call descriptor;
there is no width-only lookup shared across rows or peers. -/
inductive SelectedPlan {σ : Type} (platform : Selection σ) : Expr → Type
  | raw (x : Word) (origin : RawOrigin σ) : SelectedPlan platform (.raw x)
  | literal (x : ℝ) (site : Occurrence σ) : SelectedPlan platform (.literal x)
  | natural (x : Nat) (site : Occurrence σ) : SelectedPlan platform (.natural x)
  | negativeInfinity (site : Occurrence σ) : SelectedPlan platform .negativeInfinity
  | barrier {x : Expr} (site : Occurrence σ) (edge : Transfer) (value : SelectedPlan platform x) :
      SelectedPlan platform (.barrier x)
  | binary {a b : Expr} (op : Binary32) (site : Occurrence σ)
      (left : SelectedPlan platform a) (right : SelectedPlan platform b) : SelectedPlan platform (.binary op a b)
  | unary {x : Expr} (op : Unary32) (site : Occurrence σ) (value : SelectedPlan platform x) :
      SelectedPlan platform (.unary op x)
  | sum {width : Nat} {xs : Fin width → Expr} (choice : ReductionChoice platform .laneSum 0 width)
      (values : ∀ i, SelectedPlan platform (xs i)) : SelectedPlan platform (.sum width xs)
  | maximum {width : Nat} {xs : Fin width → Expr} (choice : ReductionChoice platform .laneMaximum 0xff800000 width)
      (values : ∀ i, SelectedPlan platform (xs i)) : SelectedPlan platform (.maximum width xs)
  | negation {x : Expr} (site : Occurrence σ) (value : SelectedPlan platform x) : SelectedPlan platform (.negation x)
  | hostAdd {a b : Expr} (site : Occurrence σ) (left : SelectedPlan platform a) (right : SelectedPlan platform b) :
      SelectedPlan platform (.hostAdd a b)
  | hostSquare {x : Expr} (site : Occurrence σ) (value : SelectedPlan platform x) : SelectedPlan platform (.hostSquare x)
  | hostScale {x : Expr} (scale : ℝ) (site : Occurrence σ) (value : SelectedPlan platform x) : SelectedPlan platform (.hostScale scale x)
  | hostSum {width : Nat} {xs : Fin width → Expr} (choice : ReductionChoice platform .torchSum 0 width)
      (values : ∀ i, SelectedPlan platform (xs i)) : SelectedPlan platform (.hostSum width xs)
  | hostLSE {width : Nat} {xs : Fin width → Expr}
      (maximum : ReductionChoice platform .torchMaximum 0xff800000 width)
      (sum : ReductionChoice platform .torchExponentialSum 0 width)
      (maskSite logSite addSite zeroSite : Occurrence σ)
      (subSite expSite : Fin width → Occurrence σ)
      (values : ∀ i, SelectedPlan platform (xs i)) : SelectedPlan platform (.hostLSE width xs)
  | rankSum {width : Nat} {xs : Fin width → Expr} (algorithm : CollectiveAlgorithm)
      (choice : ReductionChoice platform (.collective algorithm) 0 width)
      (values : ∀ i, SelectedPlan platform (xs i)) : SelectedPlan platform (.rankSum width xs)

structure LocatedCompiled (σ : Type) where
  value : LocatedGraph σ 0
  product : Option (LocatedGraph σ 0 × LocatedGraph σ 0) := none

def emitLocatedAdd (site : Occurrence σ) (a b : LocatedCompiled σ) : LocatedCompiled σ :=
  match b.product with
  | some (x,y) => ⟨.fma site x y a.value,none⟩
  | none => match a.product with
    | some (x,y) => ⟨.fma site x y b.value,none⟩
    | none => ⟨.binary site .add a.value b.value,none⟩

def emitLocatedSub (site : Occurrence σ) (a b : LocatedCompiled σ) : LocatedCompiled σ :=
  match b.product with
  | some (x,y) => ⟨.fma site (.unary site .neg x) y a.value,none⟩
  | none => ⟨.binary site .sub a.value b.value,none⟩

def hostLSELocated {σ : Type} {platform : Selection σ} {width : Nat}
    (maximum : ReductionChoice platform .torchMaximum 0xff800000 width)
    (sum : ReductionChoice platform .torchExponentialSum 0 width)
    (maskSite logSite addSite zeroSite : Occurrence σ) (subSite expSite : Fin width → Occurrence σ)
    (values : Fin width → LocatedGraph σ 0) : LocatedGraph σ 0 :=
  if width=0 then .unary logSite .hostLog (.raw zeroSite 0)
  else
    let max := LocatedGraph.unary maskSite .infinityToZero ((maximum.graph .hostMaximum).bind values)
    let exponentials := fun i => LocatedGraph.unary (expSite i) .hostExp
      (.binary (subSite i) .hostSub (values i) max)
    let total := (sum.graph .hostAdd).bind exponentials
    .binary addSite .hostAdd (.unary logSite .hostLog total) max

def SelectedPlan.compile {σ : Type} {platform : Selection σ} {expression : Expr} :
    SelectedPlan platform expression → LocatedCompiled σ
  | .raw x origin => ⟨match origin with
      | .constant site => .raw site x
      | .load site state tensor index => .load site state tensor index x,none⟩
  | .literal x site => ⟨.literal site x,none⟩
  | .natural x site => ⟨.natural site x,none⟩
  | .negativeInfinity site => ⟨.negativeInfinity site,none⟩
  | .barrier site edge x => ⟨.transfer site edge x.compile.value,none⟩
  | .binary op site a b =>
    let left := a.compile
    let right := b.compile
    match op with
    | .add => emitLocatedAdd site left right
    | .sub => emitLocatedSub site left right
    | .mul => ⟨.binary site .mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨.binary site .div left.value right.value,none⟩
  | .unary op site x => ⟨.unary site (match op with | .exp => .exp | .log => .log) x.compile.value,none⟩
  | .sum choice xs => ⟨(choice.graph .add).bind (fun i => (xs i).compile.value),none⟩
  | .maximum choice xs => ⟨(choice.graph .maximum).bind (fun i => (xs i).compile.value),none⟩
  | .negation site x => ⟨.unary site .neg x.compile.value,none⟩
  | .hostAdd site a b => ⟨.binary site .hostAdd a.compile.value b.compile.value,none⟩
  | .hostSquare site x => ⟨.unary site .hostSquare x.compile.value,none⟩
  | .hostScale scale site x => ⟨.hostScale site scale x.compile.value,none⟩
  | .hostSum choice xs => ⟨(choice.graph .hostAdd).bind (fun i => (xs i).compile.value),none⟩
  | .hostLSE maxChoice sumChoice maskSite logSite addSite zeroSite subSite expSite xs =>
    ⟨hostLSELocated maxChoice sumChoice maskSite logSite addSite zeroSite subSite expSite
      (fun i => (xs i).compile.value),none⟩
  | .rankSum _ choice xs => ⟨(choice.graph .rankAdd).bind (fun i => (xs i).compile.value),none⟩

/-- Reference interpretation of the source with this actual occurrence plan.
No native output is an input to this definition. -/
def SelectedPlan.reference {σ : Type} {platform : Selection σ} {expression : Expr}
    (plan : SelectedPlan platform expression) (reference : Numerics) : Word :=
  plan.compile.value.erase.eval reference Fin.elim0

/-- Primitive execution and scalar-port composition suffice for a complete
selected source expression; no whole-expression equality is a premise. -/
theorem selected_source_value {σ : Type} {platform : Selection σ} {expression : Expr}
    (plan : SelectedPlan platform expression) {trace : PrimitiveTrace σ} {native reference : Numerics}
    {readWord readByte} (api : PrimitiveContracts trace native readWord readByte)
    (ports : Obligations native reference) {out : Nat}
    (run : LocatedExecution trace readWord Fin.elim0 plan.compile.value out) :
    trace.bits out=plan.reference reference := run.erase.reference_value api ports

#axiomsClean selected_source_value
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
