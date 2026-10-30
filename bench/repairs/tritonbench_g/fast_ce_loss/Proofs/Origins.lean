import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.FPPlans

namespace VeriTile.Bench.Repairs.FastCeLoss.Floating
open Source
noncomputable section
set_option maxHeartbeats 3000000

structure Ref (state : Origins) where
  id : Nat
  node : Node
  earlier : id<state.next
  bound : state.nodes id=some node

def mulNode {s : Origins} (site : Site) (left right : Ref s) : Node :=
  ⟨.binary .mul left.node.expression right.node.expression,
    .binary site .mul left.node.plan right.node.plan,site,[left.id,right.id]⟩

def castNode {s : Origins} (site : Site) (value : Ref s) : Node :=
  ⟨.unary .cast value.node.expression,.unary site .cast value.node.plan,site,[value.id]⟩

inductive Tag where
  | boundary | literal | natural | pythonInt | pythonFloat | negInf | unary | binary | sum | maximum | hostSum | hostLSE
  | store | allocate

structure Event where
  site : Site
  tag : Tag
  operands : List Nat
  reads : List Address
  writes : List Address
  active : Bool

def produced (site : Site) (tag : Tag) (operands : List Nat) : Event :=
  ⟨site,tag,operands,[],[],Bool.true⟩

def Origins.clear (s : Origins) (storage : Nat) : Origins :=
  {s with cells:=fun cell => if cell.1=storage then none else s.cells cell}

/-- A source event can only consume already defined IDs. Loads use the current
reaching map. A fused subtraction must identify the actual earlier multiply
node, including its parent IDs, rather than find a matching expression later. -/
inductive OriginStep (boundary : Address → Word) : Origins → Event → Origins → Prop where
  | boundary (s : Origins) (site : Site) (cell : Address) (absent : s.cells cell=none) :
      OriginStep boundary s ⟨site,.boundary,[],[cell],[],Bool.true⟩
        ((s.push ⟨.raw (boundary cell),.raw (boundary cell),site,[]⟩).store cell s.next)
  | literal (s : Origins) (site : Site) (x : ℝ) :
      OriginStep boundary s (produced site .literal []) (s.push ⟨.literal x,.literal site x,site,[]⟩)
  | natural (s : Origins) (site : Site) (x : Nat) :
      OriginStep boundary s (produced site .natural []) (s.push ⟨.natural x,.natural site x,site,[]⟩)
  | pythonInt (s : Origins) (site : Site) (x : Int) :
      OriginStep boundary s (produced site .pythonInt []) (s.push ⟨.pythonInt x,.pythonInt site x,site,[]⟩)
  | pythonFloat (s : Origins) (site : Site) (bits : BitVec 64) :
      OriginStep boundary s (produced site .pythonFloat []) (s.push ⟨.pythonFloat bits,.pythonFloat site bits,site,[]⟩)
  | negInf (s : Origins) (site : Site) :
      OriginStep boundary s (produced site .negInf []) (s.push ⟨.negInf,.negInf site,site,[]⟩)
  | load (s : Origins) (site : Site) (cell : Address) (value : Ref s)
      (reaching : s.cells cell=some value.id) :
      OriginStep boundary s ⟨site,.unary,[value.id],[cell],[],Bool.true⟩
        (s.push ⟨.unary .load value.node.expression,.unary site .load value.node.plan,site,[value.id]⟩)
  | unary (s : Origins) (site : Site) (op : Unary) (value : Ref s) :
      OriginStep boundary s (produced site .unary [value.id])
        (s.push ⟨.unary op value.node.expression,.unary site op value.node.plan,site,[value.id]⟩)
  | binary (s : Origins) (site : Site) (op : Binary) (left right : Ref s) :
      OriginStep boundary s (produced site .binary [left.id,right.id])
        (s.push ⟨.binary op left.node.expression right.node.expression,
          .binary site op left.node.plan right.node.plan,site,[left.id,right.id]⟩)
  | mulSwap (s : Origins) (site : Site) (left right : Ref s) :
      OriginStep boundary s (produced site .binary [left.id,right.id])
        (s.push ⟨.binary .mul left.node.expression right.node.expression,
          .mulSwap site left.node.plan right.node.plan,site,[left.id,right.id]⟩)
  | fma (s : Origins) (site multiplication : Site) (swapped : Bool) (left right product addend : Ref s)
      (actualProduct : product.node=mulNode multiplication left right) :
      OriginStep boundary s (produced site .binary [product.id,addend.id])
        (s.push ⟨.binary .add (.binary .mul left.node.expression right.node.expression) addend.node.expression,
          .fma site swapped left.node.plan right.node.plan addend.node.plan,site,[product.id,addend.id]⟩)
  | fmaRight (s : Origins) (site multiplication : Site) (swapped : Bool) (addend left right product : Ref s)
      (actualProduct : product.node=mulNode multiplication left right) :
      OriginStep boundary s (produced site .binary [addend.id,product.id])
        (s.push ⟨.binary .add addend.node.expression (.binary .mul left.node.expression right.node.expression),
          .fmaRight site swapped addend.node.plan left.node.plan right.node.plan,site,[addend.id,product.id]⟩)
  | fms (s : Origins) (site multiplication : Site) (swapped : Bool) (left right product subtrahend : Ref s)
      (actualProduct : product.node=mulNode multiplication left right) :
      OriginStep boundary s (produced site .binary [product.id,subtrahend.id])
        (s.push ⟨.binary .sub (.binary .mul left.node.expression right.node.expression) subtrahend.node.expression,
          .fms site swapped left.node.plan right.node.plan subtrahend.node.plan,site,[product.id,subtrahend.id]⟩)
  | fnma (s : Origins) (site multiplication : Site) (swapped : Bool) (minuend left right product : Ref s)
      (actualProduct : product.node=mulNode multiplication left right) :
      OriginStep boundary s (produced site .binary [minuend.id,product.id])
        (s.push ⟨.binary .sub minuend.node.expression (.binary .mul left.node.expression right.node.expression),
          .fnma site swapped minuend.node.plan left.node.plan right.node.plan,site,[minuend.id,product.id]⟩)
  | fnmaCast (s : Origins) (site multiplication castSite : Site) (swapped : Bool) (minuend left right product casted : Ref s)
      (actualProduct : product.node=mulNode multiplication left right)
      (actualCast : casted.node=castNode castSite product) :
      OriginStep boundary s (produced site .binary [minuend.id,casted.id])
        (s.push ⟨.binary .sub minuend.node.expression (.unary .cast (.binary .mul left.node.expression right.node.expression)),
          .fnmaCast site castSite swapped minuend.node.plan left.node.plan right.node.plan,site,[minuend.id,casted.id]⟩)
  | fmsCast (s : Origins) (site multiplication castSite : Site) (swapped : Bool) (left right product casted subtrahend : Ref s)
      (actualProduct : product.node=mulNode multiplication left right)
      (actualCast : casted.node=castNode castSite product) :
      OriginStep boundary s (produced site .binary [casted.id,subtrahend.id])
        (s.push ⟨.binary .sub (.unary .cast (.binary .mul left.node.expression right.node.expression)) subtrahend.node.expression,
          .fmsCast site castSite swapped left.node.plan right.node.plan subtrahend.node.plan,site,[casted.id,subtrahend.id]⟩)
  | fmaCast (s : Origins) (site multiplication castSite : Site) (swapped : Bool) (left right product casted addend : Ref s)
      (actualProduct : product.node=mulNode multiplication left right)
      (actualCast : casted.node=castNode castSite product) :
      OriginStep boundary s (produced site .binary [casted.id,addend.id])
        (s.push ⟨.binary .add (.unary .cast (.binary .mul left.node.expression right.node.expression)) addend.node.expression,
          .fmaCast site castSite swapped left.node.plan right.node.plan addend.node.plan,site,[casted.id,addend.id]⟩)
  | fmaRightCast (s : Origins) (site multiplication castSite : Site) (swapped : Bool) (addend left right product casted : Ref s)
      (actualProduct : product.node=mulNode multiplication left right)
      (actualCast : casted.node=castNode castSite product) :
      OriginStep boundary s (produced site .binary [addend.id,casted.id])
        (s.push ⟨.binary .add addend.node.expression (.unary .cast (.binary .mul left.node.expression right.node.expression)),
          .fmaRightCast site castSite swapped addend.node.plan left.node.plan right.node.plan,site,[addend.id,casted.id]⟩)
  | sum (s : Origins) (site : Site) (width : Nat) (tree : Reduction width) (xs : Fin width → Ref s) :
      OriginStep boundary s (produced site .sum (List.ofFn (fun i => (xs i).id)))
        (s.push ⟨.sum width (fun i => (xs i).node.expression),.sum site tree (fun i => (xs i).node.plan),
          site,List.ofFn (fun i => (xs i).id)⟩)
  | maximum (s : Origins) (site : Site) (width : Nat) (tree : Reduction width) (xs : Fin width → Ref s) :
      OriginStep boundary s (produced site .maximum (List.ofFn (fun i => (xs i).id)))
        (s.push ⟨.maximum width (fun i => (xs i).node.expression),.maximum site tree (fun i => (xs i).node.plan),
          site,List.ofFn (fun i => (xs i).id)⟩)
  | hostSum (s : Origins) (site : Site) (width : Nat) (tree : Reduction width) (xs : Fin width → Ref s) :
      OriginStep boundary s (produced site .hostSum (List.ofFn (fun i => (xs i).id)))
        (s.push ⟨.hostSum width (fun i => (xs i).node.expression),.hostSum site tree (fun i => (xs i).node.plan),
          site,List.ofFn (fun i => (xs i).id)⟩)
  | hostLSE (s : Origins) (site : Site) (width : Nat) (maximum sum : Reduction width) (xs : Fin width → Ref s) :
      OriginStep boundary s (produced site .hostLSE (List.ofFn (fun i => (xs i).id)))
        (s.push ⟨.hostLSE width (fun i => (xs i).node.expression),.hostLSE site maximum sum (fun i => (xs i).node.plan),
          site,List.ofFn (fun i => (xs i).id)⟩)
  | store (s : Origins) (site : Site) (cell : Address) (value : Ref s) :
      OriginStep boundary s ⟨site,.store,[value.id],[],[cell],Bool.true⟩ (s.store cell value.id)
  | maskedStore (s : Origins) (site : Site) (cell : Address) (value : Ref s) :
      OriginStep boundary s ⟨site,.store,[value.id],[],[cell],Bool.false⟩ s
  | allocate (s : Origins) (site : Site) (storage : Nat) :
      OriginStep boundary s ⟨site,.allocate,[],[],[],Bool.true⟩ (s.clear storage)

inductive OriginRun (boundary : Address → Word) : Origins → List Event → Origins → Prop where
  | nil (s) : OriginRun boundary s [] s
  | cons {a b c event events} (head : OriginStep boundary a event b)
      (tail : OriginRun boundary b events c) : OriginRun boundary a (event::events) c

theorem Origins.push_wellFormed {s : Origins} (wf : s.WellFormed) (node : Node) : (s.push node).WellFormed := by
  constructor
  · intro id value found
    by_cases fresh : id=s.next
    · simp only [Origins.push]; omega
    · have prior : s.nodes id=some value := by simpa only [Origins.push,if_neg fresh] using found
      have h := wf.earlier id value prior
      simp only [Origins.push]; omega
  · intro cell id reaching
    obtain ⟨value,found⟩ := wf.reaching cell id reaching
    have different : id≠s.next := Nat.ne_of_lt (wf.earlier _ _ found)
    exact ⟨value,by simpa only [Origins.push,if_neg different] using found⟩

theorem Origins.store_wellFormed {s : Origins} (wf : s.WellFormed) (cell : Address) (id : Nat)
    (defined : ∃ node,s.nodes id=some node) : (s.store cell id).WellFormed := by
  refine ⟨wf.earlier,?_⟩
  intro target stored reaching
  by_cases same : target=cell
  · have equal : id=stored := by simpa only [Origins.store,if_pos same,Option.some.injEq] using reaching
    simpa only [←equal] using defined
  · exact wf.reaching target stored (by simpa only [Origins.store,if_neg same] using reaching)

theorem Origins.clear_wellFormed {s : Origins} (wf : s.WellFormed) (storage : Nat) : (s.clear storage).WellFormed := by
  refine ⟨wf.earlier,?_⟩
  intro cell id reaching
  by_cases same : cell.1=storage
  · simp only [Origins.clear,if_pos same] at reaching
    cases reaching
  · exact wf.reaching cell id (by simpa only [Origins.clear,if_neg same] using reaching)

theorem OriginStep.wellFormed {boundary a event b} (step : OriginStep boundary a event b)
    (wf : a.WellFormed) : b.WellFormed := by
  cases step with
  | boundary site cell absent =>
    apply Origins.store_wellFormed (Origins.push_wellFormed wf _)
    exact ⟨_,by simp only [Origins.push,ite_true]; rfl⟩
  | store site cell value => exact Origins.store_wellFormed wf cell value.id ⟨value.node,value.bound⟩
  | maskedStore => exact wf
  | allocate site storage => exact Origins.clear_wellFormed wf storage
  | _ => exact Origins.push_wellFormed wf _

theorem OriginRun.wellFormed {boundary a events b} (run : OriginRun boundary a events b)
    (wf : a.WellFormed) : b.WellFormed := by
  induction run with
  | nil => exact wf
  | cons head tail ih => exact ih (head.wellFormed wf)

#axiomsClean OriginStep.wellFormed
#axiomsClean OriginRun.wellFormed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Floating
