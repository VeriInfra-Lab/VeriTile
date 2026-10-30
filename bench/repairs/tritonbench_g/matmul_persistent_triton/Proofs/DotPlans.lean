import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.InstructionGraph

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
noncomputable section
set_option maxHeartbeats 5000000

structure Site where
  call : Nat
  astPath : List Nat
  pid : Nat
  tileOrdinal : Nat
  kIteration : Nat

/-- A selected fixed-instruction partition of one source dot. Both actual
profiles use four K fragments. Selection/order is compiler trace data; it is
not chosen by matching a mathematical output or by numerical equality. -/
structure DotLayout (width : Nat) where
  operation : ISA
  widthExact : width=operation.storage.blockK
  order : Equiv.Perm (Fin 4)

def fragmentRow (op : ISA) (row : Fin 128) (localRow : Fin op.m) : Fin 128 :=
  ⟨row.val/op.m*op.m+localRow.val,by
    have hr := row.isLt
    have hl := localRow.isLt
    cases op <;> simp only [ISA.m] at * <;> omega⟩

def fragmentColumn (op : ISA) (column : Fin 256) (localColumn : Fin op.n) : Fin 256 :=
  ⟨column.val/op.n*op.n+localColumn.val,by
    have hc := column.isLt
    have hl := localColumn.isLt
    cases op <;> simp only [ISA.n] at * <;> omega⟩

theorem k_fragment_bound (op : ISA) (chunk lane : Nat) (hc : chunk<4) (hl : lane<op.k) :
    chunk*op.k+lane<op.storage.blockK := by
  cases op <;> simp only [ISA.k,ISA.storage,Storage.blockK] at * <;> omega

theorem k_chunk_bound (op : ISA) (k : Nat) (hk : k<op.storage.blockK) : k/op.k<4 := by
  cases op <;> simp only [ISA.k,ISA.storage,Storage.blockK] at * <;> omega

def DotLayout.fragmentK {width : Nat} (layout : DotLayout width) (chunk : Fin 4)
    (lane : Fin layout.operation.k) : Fin width :=
  ⟨chunk.val*layout.operation.k+lane.val,by
    have widthExact := layout.widthExact
    have bound := k_fragment_bound layout.operation chunk.val lane.val chunk.isLt lane.isLt
    omega⟩

def localRow (op : ISA) (row : Fin 128) : Fin op.m :=
  ⟨row.val%op.m,Nat.mod_lt _ (by cases op <;> decide)⟩

def localColumn (op : ISA) (column : Fin 256) : Fin op.n :=
  ⟨column.val%op.n,Nat.mod_lt _ (by cases op <;> decide)⟩

theorem fragment_row_recover (op : ISA) (row : Fin 128) : fragmentRow op row (localRow op row)=row := by
  apply Fin.ext
  simpa only [fragmentRow,localRow,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div row.val op.m

theorem fragment_column_recover (op : ISA) (column : Fin 256) :
    fragmentColumn op column (localColumn op column)=column := by
  apply Fin.ext
  simpa only [fragmentColumn,localColumn,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div column.val op.n

/-- Every source K lane is consumed by exactly one selected instruction
fragment. The proof follows quotient/remainder, including arbitrary selected
instruction permutation inside the single source dot. -/
theorem DotLayout.k_bijective {width : Nat} (layout : DotLayout width) :
    Function.Bijective (fun pair : Fin 4 × Fin layout.operation.k =>
      layout.fragmentK (layout.order pair.1) pair.2) := by
  constructor
  · intro a b same
    have hn : 0<layout.operation.k := by cases layout.operation <;> decide
    have values := congrArg Fin.val same
    change (layout.order a.1).val*layout.operation.k+a.2.val=
      (layout.order b.1).val*layout.operation.k+b.2.val at values
    have rem := congrArg (fun n => n%layout.operation.k) values
    simp [Nat.add_mod,Nat.mod_eq_of_lt a.2.isLt,Nat.mod_eq_of_lt b.2.isLt] at rem
    have products : (layout.order a.1).val*layout.operation.k=(layout.order b.1).val*layout.operation.k := by omega
    have chunks := Nat.eq_of_mul_eq_mul_right hn products
    exact Prod.ext (layout.order.injective (Fin.ext chunks)) (Fin.ext rem)
  · intro k
    have hn : 0<layout.operation.k := by cases layout.operation <;> decide
    have hk := k.isLt
    have widthExact := layout.widthExact
    have chunkBound : k.val/layout.operation.k<4 := k_chunk_bound layout.operation k.val (by omega)
    let chunk : Fin 4 := ⟨k.val/layout.operation.k,chunkBound⟩
    let lane : Fin layout.operation.k := ⟨k.val%layout.operation.k,Nat.mod_lt _ hn⟩
    refine ⟨(layout.order.symm chunk,lane),?_⟩
    apply Fin.ext
    simpa only [DotLayout.fragmentK,Equiv.apply_symm_apply,chunk,lane,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div k.val layout.operation.k

def instructionStage {width : Nat} (layout : DotLayout width)
    (a : Fin 128 → Fin width → Graph) (b : Fin width → Fin 256 → Graph)
    (acc : Source.Acc Graph) (chunk : Fin 4) : Source.Acc Graph :=
  fun row column => .instruction layout.operation
    (fun i k => a (fragmentRow layout.operation row i) (layout.fragmentK chunk k))
    (fun k j => b (layout.fragmentK chunk k) (fragmentColumn layout.operation column j))
    (fun i j => acc (fragmentRow layout.operation row i) (fragmentColumn layout.operation column j))
    (localRow layout.operation row) (localColumn layout.operation column)

def dotGraph {width : Nat} (layout : DotLayout width)
    (a : Fin 128 → Fin width → Graph) (b : Fin width → Fin 256 → Graph)
    (acc : Source.Acc Graph) : Source.Acc Graph :=
  (List.ofFn layout.order).foldl (instructionStage layout a b) acc

/-- Each plan is indexed by exact source operands, source casts and supplied
accumulator. Only fixed ISA partitions may lower a whole-dot source marker. -/
inductive Plan : Expr → Type where
  | raw (word : Word) : Plan (.raw word)
  | zero (site : Site) : Plan .zero
  | cast (site : Site) (profile : Storage) {value : Expr} (valuePlan : Plan value) : Plan (.cast profile value)
  | dot (site : Site) {width : Nat} (layout : DotLayout width)
      {a : Fin 128 → Fin width → Expr} {b : Fin width → Fin 256 → Expr} {acc : Source.Acc Expr}
      (aPlan : ∀ i k,Plan (a i k)) (bPlan : ∀ k j,Plan (b k j)) (accPlan : ∀ i j,Plan (acc i j))
      (row : Fin 128) (column : Fin 256) : Plan (.dot width a b acc row column)

def Plan.graph : {expression : Expr} → Plan expression → Graph
  | _,.raw word => .raw word
  | _,.zero _ => .raw 0
  | _,.cast _ profile value => .cast profile value.graph
  | _,.dot _ layout a b acc row column =>
      dotGraph layout (fun i k => (a i k).graph) (fun k j => (b k j).graph)
        (fun i j => (acc i j).graph) row column

theorem Plan.equivalent {expression : Expr} (plan : Plan expression) (native reference : Numerics)
    (domain : Domain) (ports : LocalPorts native reference domain) (admitted : plan.graph.Admitted native domain) :
    plan.graph.eval native=plan.graph.eval reference := Graph.equivalent _ _ _ ports _ admitted

#axiomsClean DotLayout.k_bijective
#axiomsClean Plan.equivalent
#specNonCircular LocalPorts avoiding [Plan,Plan.graph,dotGraph,Source.accumulator]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Floating
