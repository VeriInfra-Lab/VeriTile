import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- The graph representation embeds every original scalar payload. Register
scratch, typed exterior cells, pids, numPids and undefined-load values are retained. -/
def liftState (s : RegisterState α) : RegisterState (SourceGraph.Node α) where
  mem := fun r i => mapCell SourceGraph.Node.value (s.mem r i)
  regs := fun d sh n => (s.regs d sh n).map (fun v j => mapValue SourceGraph.Node.value d (v j))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r i=>mapValue SourceGraph.Node.value d (s.undef d r i)

def rawValues (s : Shape) (o : Ops α) (initial : ScalarState α) : ChannelValues α (IO.inputs s) :=
  Fin.cases (fun j=>initial.values .a j.1.val)
    (Fin.cases (fun j=>(initial.packed .weight j.1.val).toInt)
      (Fin.cases (fun j=>initial.values .scale j.1.val)
        (Fin.cases (fun j=>(initial.packed .zero j.1.val).toInt)
          (Fin.cases (fun _=>o.zero) (fun i=>Fin.elim0 i)))))

def capacities (s : Shape) (r : RegionName) : Nat :=
  if r="a" then s.m*s.k else if r="weight" then (s.k/8)*s.n
  else if r="scale" then (s.k/s.group)*s.n else if r="zero" then (s.k/s.group)*(s.n/8)
  else if r="out" then s.m*s.n else 0

theorem raw_domain (s : Shape) (o : Ops α) (initial : ScalarState α) :
    IO.domain s o (rawValues s o initial) := by
  constructor
  · intro i
    change (BitVec.ofInt 32 ((initial.packed .weight i.val).toInt)).toInt=(initial.packed .weight i.val).toInt
    simp
  constructor
  · intro i
    change (BitVec.ofInt 32 ((initial.packed .zero i.val).toInt)).toInt=(initial.packed .zero i.val).toInt
    simp
  · intro i
    rfl

theorem raw_binding (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (initial : ScalarState α) (scratch : RegisterState α) :
    (IO.programIO x t o rest lanes).InputCells (rawValues x.shape o initial)
      (liftState (attach (zeroOutput x.shape o initial) scratch)) := by
  intro port j _
  fin_cases port
  · change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "a" j.1.val)=
      Cell.mk .real (.value (initial.values .a j.1.val))
    simp [memory,zeroOutput,mapCell,mapValue]
  · change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "weight" j.1.val)=
      Cell.mk (α:=SourceGraph.Node α) .int (initial.packed .weight j.1.val).toInt
    simp [memory,zeroOutput,mapCell,mapValue]
  · change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "scale" j.1.val)=
      Cell.mk .real (.value (initial.values .scale j.1.val))
    simp [memory,zeroOutput,mapCell,mapValue]
  · change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "zero" j.1.val)=
      Cell.mk (α:=SourceGraph.Node α) .int (initial.packed .zero j.1.val).toInt
    simp [memory,zeroOutput,mapCell,mapValue]
  · change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "out" j.1.val)=
      Cell.mk .real (.value o.zero)
    simp [memory,zeroOutput,mapCell,mapValue,j.1.isLt]

theorem raw_inBounds (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (initial : ScalarState α) :
    (IO.programIO x t o rest lanes).InBounds (capacities x.shape) (rawValues x.shape o initial) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r notBuffer
    simp only [IO.programIO,List.mem_cons,List.mem_singleton,not_or] at notBuffer
    simp only [capacities,notBuffer.1,notBuffer.2.1,notBuffer.2.2.1,notBuffer.2.2.2.1,
      notBuffer.2.2.2.2,ite_false]
  · intro port j _
    fin_cases port
    · change j.1.val<x.shape.m*x.shape.k
      exact j.1.isLt
    · change j.1.val<(x.shape.k/8)*x.shape.n
      exact j.1.isLt
    · change j.1.val<(x.shape.k/x.shape.group)*x.shape.n
      exact j.1.isLt
    · change j.1.val<(x.shape.k/x.shape.group)*(x.shape.n/8)
      exact j.1.isLt
    · change j.1.val<x.shape.m*x.shape.n
      exact j.1.isLt
  · intro port j _
    have hz : port=0 := Fin.eq_zero port
    subst port
    simpa [IO.programIO,IO.outputs,capacities] using
      rowMajor_bound x.shape.m x.shape.n j.1.val j.2.1.val j.1.isLt j.2.1.isLt
  · intro port
    exact Fin.elim0 port

def rawOutputs (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) : (IO.programIO x t o rest lanes).RawOutputs := fun key => by
  rcases key with ⟨port,j⟩
  have hz : port=0 := Fin.eq_zero port
  subst port
  exact SourceGraph.Node.value o.zero

theorem raw_output_binding (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (initial : ScalarState α) (scratch : RegisterState α) :
    (IO.programIO x t o rest lanes).OutputCells (rawValues x.shape o initial)
      (rawOutputs x t o rest lanes) (liftState (attach (zeroOutput x.shape o initial) scratch)) := by
  rintro ⟨port,j⟩ _
  have hz : port=0 := Fin.eq_zero port
  subst port
  have cap := rowMajor_bound x.shape.m x.shape.n j.1.val j.2.1.val j.1.isLt j.2.1.isLt
  change mapCell SourceGraph.Node.value (memory (zeroOutput x.shape o initial) scratch.mem "out"
    (j.1.val*x.shape.n+j.2.1.val))=Cell.mk .real (SourceGraph.Node.value o.zero)
  simp [memory,zeroOutput,mapCell,mapValue,cap]

/-- Every original arbitrary ScalarState and register scratch admits the new
finite inputs and capacities. No singleton-stride, dtype or native-word scope
is lost at the IO boundary. -/
theorem raw_admitted (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (initial : ScalarState α) (scratch : RegisterState α) :
    IO.domain x.shape o (rawValues x.shape o initial) ∧
    (IO.programIO x t o rest lanes).InputCells (rawValues x.shape o initial)
      (liftState (attach (zeroOutput x.shape o initial) scratch)) ∧
    (IO.programIO x t o rest lanes).InBounds (capacities x.shape) (rawValues x.shape o initial) ∧
    (IO.programIO x t o rest lanes).OutputCells (rawValues x.shape o initial)
      (rawOutputs x t o rest lanes) (liftState (attach (zeroOutput x.shape o initial) scratch)) :=
  ⟨raw_domain x.shape o initial,raw_binding x t o rest lanes initial scratch,
    raw_inBounds x t o rest lanes initial,raw_output_binding x t o rest lanes initial scratch⟩

#axiomsClean raw_admitted
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
