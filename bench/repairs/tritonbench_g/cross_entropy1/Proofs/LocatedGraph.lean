import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SelectedExecution

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Each scalar instruction retains the actual selected object/call and
coordinate. Erasure removes only provenance, never operand order or values. -/
inductive LocatedGraph (σ : Type) (width : Nat) where
  | input (index : Fin width)
  | load (site : Occurrence σ) (state : σ) (tensor : Tensor) (index : Nat) (bits : Word)
  | raw (site : Occurrence σ) (bits : Word)
  | literal (site : Occurrence σ) (value : ℝ)
  | natural (site : Occurrence σ) (value : Nat)
  | negativeInfinity (site : Occurrence σ)
  | binary (site : Occurrence σ) (op : ScalarBinary) (left right : LocatedGraph σ width)
  | unary (site : Occurrence σ) (op : ScalarUnary) (value : LocatedGraph σ width)
  | fma (site : Occurrence σ) (left right addend : LocatedGraph σ width)
  | hostScale (site : Occurrence σ) (scale : ℝ) (value : LocatedGraph σ width)
  | transfer (site : Occurrence σ) (edge : Transfer) (value : LocatedGraph σ width)

def LocatedGraph.erase : LocatedGraph σ width → ScalarGraph width
  | .input i => .input i
  | .load _ _ _ _ x => .raw x
  | .raw _ x => .raw x
  | .literal _ x => .literal x
  | .natural _ x => .natural x
  | .negativeInfinity _ => .negativeInfinity
  | .binary _ op a b => .binary op a.erase b.erase
  | .unary _ op a => .unary op a.erase
  | .fma _ a b c => .fma a.erase b.erase c.erase
  | .hostScale _ scale a => .hostScale scale a.erase
  | .transfer _ edge a => .transfer edge a.erase

def LocatedGraph.sites : LocatedGraph σ width → List (Occurrence σ)
  | .input _ => []
  | .load site _ _ _ _ | .raw site _ | .literal site _ | .natural site _ | .negativeInfinity site => [site]
  | .binary site _ a b => site :: (a.sites ++ b.sites)
  | .unary site _ a | .hostScale site _ a | .transfer site _ a => site :: a.sites
  | .fma site a b c => site :: (a.sites ++ b.sites ++ c.sites)

def LocatedGraph.bind (values : Fin width → LocatedGraph σ other) : LocatedGraph σ width → LocatedGraph σ other
  | .input i => values i
  | .load site state tensor index x => .load site state tensor index x
  | .raw site x => .raw site x
  | .literal site x => .literal site x
  | .natural site x => .natural site x
  | .negativeInfinity site => .negativeInfinity site
  | .binary site op a b => .binary site op (a.bind values) (b.bind values)
  | .unary site op a => .unary site op (a.bind values)
  | .fma site a b c => .fma site (a.bind values) (b.bind values) (c.bind values)
  | .hostScale site scale a => .hostScale site scale (a.bind values)
  | .transfer site edge a => .transfer site edge (a.bind values)

theorem LocatedGraph.erase_bind (values : Fin width → LocatedGraph σ other) (g : LocatedGraph σ width) :
    (g.bind values).erase=g.erase.bind (fun i => (values i).erase) := by
  induction g <;> simp_all [bind,erase,ScalarGraph.bind]

/-- Assign instruction identities inside one selected reduction. This map is
metadata only; actual execution is checked at those exact identities below. -/
structure TreeLocations {σ : Type} (site : Occurrence σ) where
  siteAt : List Nat → Occurrence σ
  sameCall : ∀ path, (siteAt path).call=site.call
  sameCoordinate : ∀ path, (siteAt path).coordinate=site.coordinate

def ReductionTree.locate (op : ScalarBinary) (sites : List Nat → Occurrence σ)
    (path : List Nat) : ReductionTree width → LocatedGraph σ width
  | .input i => .input i
  | .identity bits => .raw (sites path) bits
  | .combine a b => .binary (sites path) op
      (a.locate op sites (path++[0])) (b.locate op sites (path++[1]))
  | .transfer edge a => .transfer (sites path) edge (a.locate op sites (path++[0]))

theorem ReductionTree.erase_locate (op : ScalarBinary) (sites : List Nat → Occurrence σ)
    (path : List Nat) (tree : ReductionTree width) :
    (tree.locate op sites path).erase=tree.graph op := by
  induction tree generalizing path <;> simp_all [locate,LocatedGraph.erase,graph]

/-- Located execution prevents using a selected tree from one invocation
with arithmetic or communication events from an unrelated invocation. -/
inductive LocatedExecution {σ : Type} (trace : PrimitiveTrace σ)
    (readWord : σ → Tensor → Nat → Word) (inputs : Fin width → Word) :
    LocatedGraph σ width → Nat → Prop
  | input (i : Fin width) (site state tensor index out)
      (event : trace.load site state tensor index out)
      (binding : readWord state tensor index=inputs i) : LocatedExecution trace readWord inputs (.input i) out
  | loaded (site state tensor index x out)
      (event : trace.load site state tensor index out)
      (binding : readWord state tensor index=x) : LocatedExecution trace readWord inputs (.load site state tensor index x) out
  | raw (site x out) (event : trace.raw site x out) : LocatedExecution trace readWord inputs (.raw site x) out
  | literal (site x out) (event : trace.literal site x out) : LocatedExecution trace readWord inputs (.literal site x) out
  | natural (site x out) (event : trace.natural site x out) : LocatedExecution trace readWord inputs (.natural site x) out
  | negativeInfinity (site out) (event : trace.negativeInfinity site out) :
      LocatedExecution trace readWord inputs (.negativeInfinity site) out
  | binary (site op a b av bv out) (left : LocatedExecution trace readWord inputs a av)
      (right : LocatedExecution trace readWord inputs b bv) (event : trace.binary site op av bv out) :
      LocatedExecution trace readWord inputs (.binary site op a b) out
  | unary (site op a av out) (value : LocatedExecution trace readWord inputs a av)
      (event : trace.unary site op av out) : LocatedExecution trace readWord inputs (.unary site op a) out
  | fma (site a b c av bv cv out) (left : LocatedExecution trace readWord inputs a av)
      (right : LocatedExecution trace readWord inputs b bv) (addend : LocatedExecution trace readWord inputs c cv)
      (event : trace.fma site av bv cv out) : LocatedExecution trace readWord inputs (.fma site a b c) out
  | hostScale (site scale a av out) (value : LocatedExecution trace readWord inputs a av)
      (event : trace.hostScale site scale av out) : LocatedExecution trace readWord inputs (.hostScale site scale a) out
  | transfer (site edge a av out) (value : LocatedExecution trace readWord inputs a av)
      (event : trace.transfer site edge av out) : LocatedExecution trace readWord inputs (.transfer site edge a) out

theorem LocatedExecution.erase {σ : Type} {trace : PrimitiveTrace σ}
    {readWord inputs} {graph : LocatedGraph σ width} {out : Nat}
    (run : LocatedExecution trace readWord inputs graph out) :
    Execution trace readWord inputs graph.erase out := by
  induction run with
  | input i site state tensor index out event binding => exact .input i site state tensor index out event binding
  | loaded site state tensor index x out event binding => exact .loaded site state tensor index x out event binding
  | raw site x out event => exact .raw site x out event
  | literal site x out event => exact .literal site x out event
  | natural site x out event => exact .natural site x out event
  | negativeInfinity site out event => exact .negativeInfinity site out event
  | binary site op a b av bv out _ _ event ihA ihB => exact .binary site op a.erase b.erase av bv out ihA ihB event
  | unary site op a av out _ event ih => exact .unary site op a.erase av out ih event
  | fma site a b c av bv cv out _ _ _ event ihA ihB ihC => exact .fma site a.erase b.erase c.erase av bv cv out ihA ihB ihC event
  | hostScale site scale a av out _ event ih => exact .hostScale site scale a.erase av out ih event
  | transfer site edge a av out _ event ih => exact .transfer site edge a.erase av out ih event

#axiomsClean LocatedGraph.erase_bind
#axiomsClean ReductionTree.erase_locate
#axiomsClean LocatedExecution.erase
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
