import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.DotPlans

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
noncomputable section
set_option maxHeartbeats 5000000

structure Node where
  expression : Expr
  plan : Plan expression
  site : Site
  parents : List Nat

def Node.transport (node : Node) {expression : Expr} (same : node.expression=expression) : Plan expression :=
  same ▸ node.plan

theorem Node.transport_graph (node : Node) {expression : Expr} (same : node.expression=expression) :
    (node.transport same).graph=node.plan.graph := by cases same; rfl

structure Origins where
  next : Nat
  nodes : Nat → Option Node
  cells : Address → Option Nat

def Origins.push (state : Origins) (node : Node) : Origins :=
  ⟨state.next+1,fun id => if id=state.next then some node else state.nodes id,state.cells⟩

def Origins.store (state : Origins) (cell : Address) (id : Nat) : Origins :=
  {state with cells:=fun point => if point=cell then some id else state.cells point}

def Origins.clear (state : Origins) (storage : Nat) : Origins :=
  {state with cells:=fun point => if point.1=storage then none else state.cells point}

def emptyOrigins : Origins := ⟨0,fun _ => none,fun _ => none⟩

structure Ref (state : Origins) where
  id : Nat
  node : Node
  earlier : id<state.next
  found : state.nodes id=some node

inductive Tag where
  | boundary | load | zero | cast | leaky | dot | store | allocate

structure Event where
  site : Site
  tag : Tag
  parents : List Nat
  reads : List Address
  writes : List Address
  active : Bool

def matrixIDs {state : Origins} {rows columns : Nat} (values : Fin rows → Fin columns → Ref state) : List Nat :=
  (List.ofFn (fun row => List.ofFn (fun column => (values row column).id))).flatten

inductive OriginStep (boundary : Address → Word) : Origins → Event → Origins → Prop where
  | boundary (s : Origins) (site : Site) (cell : Address) (absent : s.cells cell=none) :
      OriginStep boundary s ⟨site,.boundary,[],[cell],[],Bool.true⟩
        ((s.push ⟨.raw (boundary cell),.raw (boundary cell),site,[]⟩).store cell s.next)
  | load (s : Origins) (site : Site) (cell : Address) (value : Ref s) (reaching : s.cells cell=some value.id) :
      OriginStep boundary s ⟨site,.load,[value.id],[cell],[],Bool.true⟩
        (s.push ⟨value.node.expression,value.node.plan,site,[value.id]⟩)
  | zero (s : Origins) (site : Site) :
      OriginStep boundary s ⟨site,.zero,[],[],[],Bool.true⟩ (s.push ⟨.zero,.zero site,site,[]⟩)
  | cast (s : Origins) (site : Site) (value : Ref s) :
      OriginStep boundary s ⟨site,.cast,[value.id],[],[],Bool.true⟩
        (s.push ⟨.cast value.node.expression,.cast site value.node.plan,site,[value.id]⟩)
  | leaky (s : Origins) (site : Site) (swappedMultiply : Bool) (value : Ref s) :
      OriginStep boundary s ⟨site,.leaky,[value.id],[],[],Bool.true⟩
        (s.push ⟨.leaky value.node.expression,.leaky site swappedMultiply value.node.plan,site,[value.id]⟩)
  | dot (s : Origins) (site : Site) (width : Nat) (layout : DotLayout width)
      (a : Fin 128 → Fin width → Ref s) (b : Fin width → Fin 256 → Ref s)
      (acc : Source.Acc (Ref s)) (row : Fin 128) (column : Fin 256) :
      OriginStep boundary s ⟨site,.dot,matrixIDs a++matrixIDs b++matrixIDs acc,[],[],Bool.true⟩
        (s.push ⟨.dot width (fun i k => (a i k).node.expression) (fun k j => (b k j).node.expression)
          (fun i j => (acc i j).node.expression) row column,
          .dot site layout (fun i k => (a i k).node.plan) (fun k j => (b k j).node.plan)
            (fun i j => (acc i j).node.plan) row column,site,matrixIDs a++matrixIDs b++matrixIDs acc⟩)
  | store (s : Origins) (site : Site) (cell : Address) (value : Ref s) :
      OriginStep boundary s ⟨site,.store,[value.id],[],[cell],Bool.true⟩ (s.store cell value.id)
  | maskedStore (s : Origins) (site : Site) (cell : Address) (value : Ref s) :
      OriginStep boundary s ⟨site,.store,[value.id],[],[cell],Bool.false⟩ s
  | allocate (s : Origins) (site : Site) (storage : Nat) :
      OriginStep boundary s ⟨site,.allocate,[],[],[],Bool.true⟩ (s.clear storage)

inductive OriginRun (boundary : Address → Word) : Origins → List Event → Origins → Prop where
  | nil (state) : OriginRun boundary state [] state
  | cons {a b c event events} (head : OriginStep boundary a event b) (tail : OriginRun boundary b events c) :
      OriginRun boundary a (event::events) c

structure Origins.WellFormed (state : Origins) : Prop where
  earlier : ∀ id node,state.nodes id=some node → id<state.next
  reaching : ∀ cell id,state.cells cell=some id → ∃ node,state.nodes id=some node

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
  | store site cell value => exact Origins.store_wellFormed wf cell value.id ⟨value.node,value.found⟩
  | maskedStore => exact wf
  | allocate site storage => exact Origins.clear_wellFormed wf storage
  | _ => exact Origins.push_wellFormed wf _

theorem OriginRun.wellFormed {boundary a events b} (run : OriginRun boundary a events b)
    (wf : a.WellFormed) : b.WellFormed := by
  induction run with
  | nil => exact wf
  | cons head tail ih => exact ih (head.wellFormed wf)

#axiomsClean OriginRun.wellFormed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
