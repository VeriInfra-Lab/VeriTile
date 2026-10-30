import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SignedValues

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident.Origins
open VeriTile.Triton
open CrossEntropy1.Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Dynamic source identity: the AST path and visit distinguish repeated
occurrences even when their erased arithmetic expressions are equal. -/
structure Point (σ : Type) where
  astPath : List Nat
  visit : Nat
  site : Occurrence σ

structure Value {σ : Type} (selection : Selection σ) where
  expression : Expr
  plan : SelectedPlan selection expression

abbrev Address := RegionName × Nat

structure Stored where
  id : Nat
  negative : Bool

/-- SSA ids are append-only. The cell map is the current reaching definition,
not an existential choice made after seeing an output word. -/
structure State {σ : Type} (selection : Selection σ) where
  next : Nat
  nodes : Nat → Option (Value selection)
  cells : Address → Option Stored

structure State.WellFormed {σ : Type} {selection : Selection σ} (s : State selection) : Prop where
  earlier : ∀ id node, s.nodes id=some node → id<s.next
  reaching : ∀ address stored, s.cells address=some stored → ∃ node, s.nodes stored.id=some node

structure Ref {σ : Type} {selection : Selection σ} (s : State selection) where
  id : Nat
  value : Value selection
  earlier : id<s.next
  bound : s.nodes id=some value

structure Operand {σ : Type} {selection : Selection σ} (s : State selection) extends Ref s where
  negative : Bool

def Operand.stored {σ : Type} {selection : Selection σ} {s : State selection} (v : Operand s) : Stored :=
  ⟨v.id,v.negative⟩

def Operand.expression {σ : Type} {selection : Selection σ} {s : State selection} (v : Operand s) : Expr :=
  if v.negative then .negation v.value.expression else v.value.expression

def Operand.plan {σ : Type} {selection : Selection σ} {s : State selection} (v : Operand s)
    (site : Occurrence σ) : SelectedPlan selection v.expression :=
  if h : v.negative then by
    simpa only [Operand.expression,h,ite_true] using SelectedPlan.negation site v.value.plan
  else by simpa only [Operand.expression,h,ite_false] using v.value.plan

def State.push {σ : Type} {selection : Selection σ} (s : State selection) (v : Value selection) : State selection :=
  {s with next:=s.next+1,nodes:=fun i => if i=s.next then some v else s.nodes i}

def State.store {σ : Type} {selection : Selection σ} (s : State selection) (address : Address) (value : Stored) : State selection :=
  {s with cells:=fun cell => if cell=address then some value else s.cells cell}

def State.clear {σ : Type} {selection : Selection σ} (s : State selection) (region : RegionName) : State selection :=
  {s with cells:=fun cell => if cell.1=region then none else s.cells cell}

theorem State.push_wellFormed {σ : Type} {selection : Selection σ} {s : State selection}
    (wf : s.WellFormed) (v : Value selection) : (s.push v).WellFormed := by
  constructor
  · intro id node bound
    by_cases fresh : id=s.next
    · simp only [State.push]; omega
    · have prior : s.nodes id=some node := by simpa only [State.push,if_neg fresh] using bound
      have h := wf.earlier id node prior
      simp only [State.push]; omega
  · intro address stored atCell
    obtain ⟨node,bound⟩ := wf.reaching address stored atCell
    have different : stored.id≠s.next := Nat.ne_of_lt (wf.earlier _ _ bound)
    exact ⟨node,by simpa only [State.push,if_neg different] using bound⟩

theorem State.store_wellFormed {σ : Type} {selection : Selection σ} {s : State selection}
    (wf : s.WellFormed) (address : Address) (stored : Stored)
    (defined : ∃ node, s.nodes stored.id=some node) : (s.store address stored).WellFormed := by
  refine ⟨wf.earlier,?_⟩
  intro cell value atCell
  by_cases same : cell=address
  · have equal : stored=value := by simpa only [State.store,if_pos same,Option.some.injEq] using atCell
    simpa only [←equal] using defined
  · exact wf.reaching cell value (by simpa only [State.store,if_neg same] using atCell)

theorem State.clear_wellFormed {σ : Type} {selection : Selection σ} {s : State selection}
    (wf : s.WellFormed) (region : RegionName) : (s.clear region).WellFormed := by
  refine ⟨wf.earlier,?_⟩
  intro cell stored atCell
  by_cases same : cell.1=region
  · simp only [State.clear,if_pos same] at atCell
    cases atCell
  · exact wf.reaching cell stored (by simpa only [State.clear,if_neg same] using atCell)

inductive Tag where
  | boundary | literal | natural | negativeInfinity | load | barrier
  | binary | unary | sum | maximum | negation
  | hostAdd | hostSquare | hostScale | hostSum | hostLSE | rankSum
  | store | copy | allocate

structure Event (σ : Type) where
  point : Point σ
  tag : Tag
  operands : List Nat
  reads : List Address
  writes : List Address
  active : Bool

def produced {σ : Type} (point : Point σ) (tag : Tag) (operands : List Nat) : Event σ :=
  ⟨point,tag,operands,[],[],Bool.true⟩

/-- A source-level step, not a native-opcode certificate. Every arithmetic
constructor reuses the actual previously defined child plans. A load reads
the current cell map. Copy migrates its id without adding an Expr barrier.
Only a source store whose mask is active updates the reaching definition.
Allocation clears all coordinates, including holes. -/
inductive Step {σ : Type} (selection : Selection σ)
    (boundaryWord : σ → RegionName → Nat → Word) : State selection → Event σ → State selection → Prop
  | boundary (s : State selection) (p : Point σ) (native : σ) (tensor : Tensor) (index : Nat)
      (absent : s.cells (tensor.region,tensor.index index)=none)
      (actualBefore : native=p.site.call.before) (input : tensor∈p.site.call.inputs) (inside : index<tensor.numel) :
      Step selection boundaryWord s ⟨p,.boundary,[],[(tensor.region,tensor.index index)],[],Bool.true⟩
        ((s.push ⟨.raw (boundaryWord native tensor.region (tensor.index index)),
          .raw _ (.load p.site native {tensor with isNeg:=Bool.false} index)⟩).store
            (tensor.region,tensor.index index) ⟨s.next,Bool.false⟩)
  | literal (s : State selection) (p : Point σ) (x : ℝ) :
      Step selection boundaryWord s (produced p .literal []) (s.push ⟨.literal x,.literal x p.site⟩)
  | natural (s : State selection) (p : Point σ) (x : Nat) :
      Step selection boundaryWord s (produced p .natural []) (s.push ⟨.natural x,.natural x p.site⟩)
  | negativeInfinity (s : State selection) (p : Point σ) :
      Step selection boundaryWord s (produced p .negativeInfinity []) (s.push ⟨.negativeInfinity,.negativeInfinity p.site⟩)
  | load (s : State selection) (p : Point σ) (address : Address) (v : Operand s) (edge : Transfer)
      (reaching : s.cells address=some v.stored) :
      Step selection boundaryWord s ⟨p,.load,[v.id],[address],[],Bool.true⟩
        (s.push ⟨.barrier v.expression,.barrier p.site edge (v.plan p.site)⟩)
  | barrier (s : State selection) (p : Point σ) (v : Operand s) (edge : Transfer) :
      Step selection boundaryWord s (produced p .barrier [v.id])
        (s.push ⟨.barrier v.expression,.barrier p.site edge (v.plan p.site)⟩)
  | binary (s : State selection) (p : Point σ) (op : CrossEntropy1.Source.Binary32) (a b : Operand s) :
      Step selection boundaryWord s (produced p .binary [a.id,b.id])
        (s.push ⟨.binary op a.expression b.expression,.binary op p.site (a.plan p.site) (b.plan p.site)⟩)
  | unary (s : State selection) (p : Point σ) (op : CrossEntropy1.Source.Unary32) (v : Operand s) :
      Step selection boundaryWord s (produced p .unary [v.id])
        (s.push ⟨.unary op v.expression,.unary op p.site (v.plan p.site)⟩)
  | sum (s : State selection) (p : Point σ) (width : Nat) (vs : Fin width → Operand s)
      (choice : ReductionChoice selection .laneSum 0 width) (same : choice.selected.site=p.site) :
      Step selection boundaryWord s (produced p .sum (List.ofFn (fun i => (vs i).id)))
        (s.push ⟨.sum width (fun i => (vs i).expression),.sum choice (fun i => ((vs i).plan p.site))⟩)
  | maximum (s : State selection) (p : Point σ) (width : Nat) (vs : Fin width → Operand s)
      (choice : ReductionChoice selection .laneMaximum 0xff800000 width) (same : choice.selected.site=p.site) :
      Step selection boundaryWord s (produced p .maximum (List.ofFn (fun i => (vs i).id)))
        (s.push ⟨.maximum width (fun i => (vs i).expression),.maximum choice (fun i => ((vs i).plan p.site))⟩)
  | negation (s : State selection) (p : Point σ) (v : Operand s) :
      Step selection boundaryWord s (produced p .negation [v.id])
        (s.push ⟨.negation v.expression,.negation p.site (v.plan p.site)⟩)
  | hostAdd (s : State selection) (p : Point σ) (a b : Operand s) :
      Step selection boundaryWord s (produced p .hostAdd [a.id,b.id])
        (s.push ⟨.hostAdd a.expression b.expression,.hostAdd p.site (a.plan p.site) (b.plan p.site)⟩)
  | hostSquare (s : State selection) (p : Point σ) (v : Operand s) :
      Step selection boundaryWord s (produced p .hostSquare [v.id])
        (s.push ⟨.hostSquare v.expression,.hostSquare p.site (v.plan p.site)⟩)
  | hostScale (s : State selection) (p : Point σ) (scale : ℝ) (v : Operand s) :
      Step selection boundaryWord s (produced p .hostScale [v.id])
        (s.push ⟨.hostScale scale v.expression,.hostScale scale p.site (v.plan p.site)⟩)
  | hostSum (s : State selection) (p : Point σ) (width : Nat) (vs : Fin width → Operand s)
      (choice : ReductionChoice selection .torchSum 0 width) (same : choice.selected.site=p.site) :
      Step selection boundaryWord s (produced p .hostSum (List.ofFn (fun i => (vs i).id)))
        (s.push ⟨.hostSum width (fun i => (vs i).expression),.hostSum choice (fun i => ((vs i).plan p.site))⟩)
  | hostLSE (s : State selection) (p : Point σ) (width : Nat) (vs : Fin width → Operand s)
      (maxChoice : ReductionChoice selection .torchMaximum 0xff800000 width)
      (sumChoice : ReductionChoice selection .torchExponentialSum 0 width)
      (maskSite logSite addSite zeroSite : Occurrence σ) (subSite expSite : Fin width → Occurrence σ) :
      Step selection boundaryWord s (produced p .hostLSE (List.ofFn (fun i => (vs i).id)))
        (s.push ⟨.hostLSE width (fun i => (vs i).expression),
          .hostLSE maxChoice sumChoice maskSite logSite addSite zeroSite subSite expSite (fun i => ((vs i).plan p.site))⟩)
  | rankSum (s : State selection) (p : Point σ) (width : Nat) (vs : Fin width → Operand s)
      (algorithm : CollectiveAlgorithm) (choice : ReductionChoice selection (.collective algorithm) 0 width)
      (same : choice.selected.site=p.site) :
      Step selection boundaryWord s (produced p .rankSum (List.ofFn (fun i => (vs i).id)))
        (s.push ⟨.rankSum width (fun i => (vs i).expression),.rankSum algorithm choice (fun i => ((vs i).plan p.site))⟩)
  | store (s : State selection) (p : Point σ) (address : Address) (v : Operand s) :
      Step selection boundaryWord s ⟨p,.store,[v.id],[],[address],Bool.true⟩ (s.store address v.stored)
  | maskedStore (s : State selection) (p : Point σ) (address : Address) (v : Operand s) :
      Step selection boundaryWord s ⟨p,.store,[v.id],[],[address],Bool.false⟩ s
  | copy (s : State selection) (p : Point σ) (src dst : Tensor) (index : Nat) (v : Operand s)
      (reaching : s.cells (src.region,src.index index)=some v.stored)
      (input : src∈p.site.call.inputs) (output : dst∈p.site.call.outputs)
      (insideSrc : index<src.numel) (insideDst : index<dst.numel) :
      Step selection boundaryWord s ⟨p,.copy,[v.id],[(src.region,src.index index)],[(dst.region,dst.index index)],Bool.true⟩
        (s.store (dst.region,dst.index index) ⟨v.id,Bool.xor (Bool.xor v.negative src.isNeg) dst.isNeg⟩)
  | allocate (s : State selection) (p : Point σ) (region : RegionName) :
      Step selection boundaryWord s ⟨p,.allocate,[],[],[(region,0)],Bool.true⟩ (s.clear region)

inductive Run {σ : Type} (selection : Selection σ) (boundaryWord : σ → RegionName → Nat → Word) :
    State selection → List (Event σ) → State selection → Prop
  | nil (s) : Run selection boundaryWord s [] s
  | cons {a b c event events} (head : Step selection boundaryWord a event b)
      (tail : Run selection boundaryWord b events c) : Run selection boundaryWord a (event::events) c

theorem Step.wellFormed {σ : Type} {selection : Selection σ} {readRaw a event b}
    (step : Step selection readRaw a event b) (wf : a.WellFormed) : b.WellFormed := by
  cases step with
  | boundary p native tensor index absent actualBefore input inside =>
    apply State.store_wellFormed (State.push_wellFormed wf _)
    exact ⟨_,by simp only [State.push,ite_true]; rfl⟩
  | store p address v => exact State.store_wellFormed wf address v.stored ⟨v.value,v.bound⟩
  | maskedStore => exact wf
  | copy p src dst index v reaching input output insideSrc insideDst =>
    exact State.store_wellFormed wf _ _ ⟨v.value,v.bound⟩
  | allocate p region => exact State.clear_wellFormed wf region
  | _ => exact State.push_wellFormed wf _

theorem Run.wellFormed {σ : Type} {selection : Selection σ} {readRaw a events b}
    (run : Run selection readRaw a events b) (wf : a.WellFormed) : b.WellFormed := by
  induction run with
  | nil => exact wf
  | cons head tail ih => exact ih (head.wellFormed wf)

/-- Reindexing never reselects a reduction, occurrence, or fusion policy. -/
def Value.transport {σ : Type} {selection : Selection σ} (v : Value selection)
    {expression : Expr} (same : v.expression=expression) : SelectedPlan selection expression :=
  same ▸ v.plan

theorem Value.transport_reference {σ : Type} {selection : Selection σ} (v : Value selection)
    {expression : Expr} (same : v.expression=expression) (contract : Occurrence σ → Bool) (numerics : Numerics) :
    ConfiguredPlan.reference contract (v.transport same) numerics=
      ConfiguredPlan.reference contract v.plan numerics := by cases same; rfl

theorem copy_keeps_plan {σ : Type} {selection : Selection σ} (s : State selection)
    (dst : Address) (v : Operand s) (srcSign dstSign : Bool) :
    (s.store dst ⟨v.id,Bool.xor (Bool.xor v.negative srcSign) dstSign⟩).cells dst=
        some ⟨v.id,Bool.xor (Bool.xor v.negative srcSign) dstSign⟩ ∧
      (s.store dst ⟨v.id,Bool.xor (Bool.xor v.negative srcSign) dstSign⟩).nodes v.id=some v.value := by
  exact ⟨by simp [State.store],v.bound⟩

theorem allocate_clears_holes {σ : Type} {selection : Selection σ} (s : State selection)
    (region : RegionName) (index : Nat) : (s.clear region).cells (region,index)=none := by
  simp [State.clear]

#axiomsClean Value.transport_reference
#axiomsClean Run.wellFormed
#axiomsClean copy_keeps_plan
#axiomsClean allocate_clears_holes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident.Origins
