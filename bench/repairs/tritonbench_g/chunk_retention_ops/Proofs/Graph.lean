import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Transport
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.SeparateAPI
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

/-- Value-independent scalar instructions. Load leaves name source logical
addresses; no input word or expected result occurs in the compiler graph. -/
inductive Term where
  | load : Address → Term
  | scale : Term
  | zero : Term
  | integer : Int → Term
  | cast : Term → Term
  | add : Term → Term → Term
  | sub : Term → Term → Term
  | mul : Term → Term → Term
  | fma : Term → Term → Term → Term
  | exp2 : Term → Term
  | log2 : Term → Term

def termOps : Ops Term := ⟨.zero,.integer,.cast,.add,.sub,.mul,.fma,.exp2,.log2⟩
def Term.eval (ops : Ops α) (scale : α) (m : Heap α) : Term → α
  | .load a => m a
  | .scale => scale
  | .zero => ops.zero
  | .integer n => ops.integer n
  | .cast x => ops.cast (x.eval ops scale m)
  | .add x y => ops.add (x.eval ops scale m) (y.eval ops scale m)
  | .sub x y => ops.sub (x.eval ops scale m) (y.eval ops scale m)
  | .mul x y => ops.mul (x.eval ops scale m) (y.eval ops scale m)
  | .fma x y z => ops.fma (x.eval ops scale m) (y.eval ops scale m) (z.eval ops scale m)
  | .exp2 x => ops.exp2 (x.eval ops scale m)
  | .log2 x => ops.log2 (x.eval ops scale m)

theorem term_hom (ops : Ops α) (scale : α) (m : Heap α) :
    Ops.Hom termOps ops (Term.eval ops scale m) := by
  constructor <;> intros <;> rfl

end Checked
namespace Ops.Hom
variable {a : Ops α} {b : Ops β} {f : α → β} (hom : Ops.Hom a b f)
include hom

theorem body_map (c : Shape) (scale : α) (tree : SumTree) (p : Phase) (m : Heap α) (x : Address) :
    f (Source.bodyRHS a c scale tree p m x)=
      Source.bodyRHS b c (f scale) tree p (fun z => f (m z)) x := by
  cases p <;> simp only [Source.bodyRHS]
  · split_ifs
    · rw [hom.cast]
      have h := congrFun (congrFun (hom.history_map c (inputsOfHeap m) x.head x.slice) x.row) x.col
      rw [h]
      rfl
    · unfold Reference.finalState
      rw [hom.cast]
      have h := congrFun (congrFun (hom.history_map c (inputsOfHeap m) x.head c.chunks) x.row) x.col
      rw [h]
      rfl
  · rw [hom.output_map]
    rfl
  · split_ifs
    · rw [hom.cast]
      have h := congrFun (congrFun (hom.dhistory_map c (inputsOfHeap m) scale x.head x.slice) x.row) x.col
      rw [h]
      rfl
    · rw [hom.cast]
      have h := congrFun (congrFun (hom.reverse_map c (inputsOfHeap m) scale x.head c.chunks) x.row) x.col
      rw [h]
      rfl
  · split_ifs
    · rw [hom.dq_map]
      rfl
    · rw [hom.dk_map]
      rfl
    · rw [hom.partial_map]
      rfl
  · exact hom.sum_map _ _

theorem project_map (p : Phase) (m : Heap α) :
    (fun x => f (project a p m x))=project b p (fun x => f (m x)) := by
  funext x
  unfold project
  split_ifs
  · rfl
  · exact hom.zero

theorem rhs_map (c : Shape) (scale : α) (tree : SumTree) (p : Phase) (m : Heap α) (x : Address) :
    f (Source.rhs a c scale tree p m x)=
      Source.rhs b c (f scale) tree p (fun z => f (m z)) x := by
  unfold Source.rhs
  rw [hom.body_map,hom.project_map]
end Ops.Hom
namespace Checked

structure Store where
  address : Address
  rhs : Term

def phaseCode (c : Shape) (tree : SumTree) (p : Phase) (events : List Address) : List Store :=
  events.map fun a => ⟨a,Source.rhs termOps c .scale tree p Term.load a⟩

def forwardCode (c : Shape) (tree : SumTree) (rf : Bool) : List Store :=
  phaseCode c tree .h (Source.eventsH c rf)++phaseCode c tree .o (Source.eventsO c)
def backwardCode (c : Shape) (tree : SumTree) (ri : Bool) : List Store :=
  phaseCode c tree .dh (Source.eventsDH c++if ri then Source.eventsInitial c else [])++
  phaseCode c tree .qkv (Source.eventsQKV c)++phaseCode c tree .sum (Source.eventsSum c)

/-- Every instruction reads the memory returned by its predecessor. -/
def execute (ops : Ops α) (c : Shape) (b : Binding c) (scale : α) : List Store → Memory α → Memory α
  | [],m => m
  | s::rest,m => execute ops c b scale rest
      (physicalWrite m (b.address s.address) (s.rhs.eval ops scale (b.view ops.zero m)))

theorem execute_append (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (xs ys : List Store) (m : Memory α) :
    execute ops c b scale (xs++ys) m=execute ops c b scale ys (execute ops c b scale xs m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xs ih => exact ih _

theorem execute_phase (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (p : Phase) (events : List Address) (m : Memory α) :
    execute ops c b scale (phaseCode c tree p events) m=
      Source.executePhysical ops c b scale tree p events m := by
  induction events generalizing m with
  | nil => rfl
  | cons x xs ih =>
    simp only [phaseCode,List.map_cons,execute,Source.executePhysical]
    have h := (term_hom ops scale (b.view ops.zero m)).rhs_map c Term.scale tree p Term.load x
    change (Source.rhs termOps c Term.scale tree p Term.load x).eval ops scale (b.view ops.zero m)=
      Source.rhs ops c scale tree p (b.view ops.zero m) x at h
    rw [h]
    exact ih _

theorem forward_graph (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (rf : Bool) (m : Memory α) :
    execute ops c b scale (forwardCode c tree rf) m=Separate.forwardPhysical ops c b scale tree rf m := by
  simp only [forwardCode,execute_append,execute_phase,Separate.forwardPhysical]

theorem backward_graph (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (tree : SumTree) (ri : Bool) (m : Memory α) :
    execute ops c b scale (backwardCode c tree ri) m=Separate.backwardPhysical ops c b scale tree ri m := by
  simp only [backwardCode,execute_append,execute_phase,Separate.backwardPhysical]

#axiomsClean forward_graph
#axiomsClean backward_graph
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
