import bench.repairs.tritonbench_g.triton_softmax.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.TritonSoftmax.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Loads embed a physical word as the same value in every possible thread.
The source carrier is temporary; it is never exposed as a physical byte. -/
def liftState (s : State Word) : State Node where
  mem := fun r i => .mk .real (atom ((s.mem r i).read .real))
  regs := fun _ _ _ => none
  pids := s.pids
  numPids := s.numPids
  undef := fun d _ _ => defaultValue d

/-- Each stored output is read at the owning thread of its logical column.
Off-output cells keep their original full Cell, including non-real tags. -/
def projectState (V B : Nat) (y : Tensor) (p : Nat) (s : State Word) (t : State Node) : State Word :=
  {s with mem := fun r i =>
    if h : ∃ c, c<y.cols ∧ r=y.region ∧ i=y.index p c then
      Cell.mk TileDType.real (((t.mem r i).read .real) (owner V B (Nat.find h)))
    else s.mem r i}

def evaluate (inst : Instructions) (V : Nat) (x y : Tensor) (B p : Nat)
    (s : State Word) : Option (State Word) :=
  (Floating.evaluate (algebra inst V) x y B p (liftState s)).map (projectState V B y p s)

theorem project_active (V B : Nat) (y : Tensor) (p c : Nat)
    (hp : p<y.rows) (hc : c<y.cols) (inj : y.Injective) (s : State Word) (t : State Node) :
    (projectState V B y p s t).mem y.region (y.index p c)=
      .mk .real (((t.mem y.region (y.index p c)).read .real) (owner V B c)) := by
  have h : ∃ j, j<y.cols ∧ y.region=y.region ∧ y.index p c=y.index p j :=
    ⟨c,hc,rfl,rfl⟩
  have chosen : Nat.find h=c := by
    have spec := Nat.find_spec h
    exact (inj p (Nat.find h) p c hp spec.1 hp hc spec.2.2.symm).2
  change (if h' : ∃ j,j<y.cols ∧ y.region=y.region ∧ y.index p c=y.index p j then
    Cell.mk TileDType.real (((t.mem y.region (y.index p c)).read .real) (owner V B (Nat.find h')))
    else s.mem y.region (y.index p c))=_
  rw [dif_pos h,chosen]

theorem source (inst : Instructions) (V : Nat) :
    SourceContract Floating.access (evaluate inst V) (row inst V) := by
  constructor
  intro x y B p s t values shape inj hB cover hp input run
  cases he : Floating.evaluate (algebra inst V) x y B p (liftState s) with
  | none => simp [evaluate,he] at run
  | some nodes =>
    have eq : projectState V B y p s nodes=t := by simpa [evaluate,he] using run
    subst t
    have nodeInput : Inputs Floating.access x (fun r c => atom (values r c)) (liftState s) := by
      intro r c hr hc
      change atom ((s.mem x.region (x.index r c)).read .real)=atom (values r c)
      exact congrArg atom (input r c hr hc)
    have src := (Floating.source (algebra inst V)).row x y B p (liftState s) nodes
      (fun r c => atom (values r c)) shape inj hB cover hp nodeInput he
    constructor
    · intro c hc
      change ((projectState V B y p s nodes).mem y.region (y.index p c)).read .real=_
      rw [project_active V B y p c (by rwa [shape.rows]) (by rwa [shape.cols]) inj]
      change ((nodes.mem y.region (y.index p c)).read .real) (owner V B c)=_
      have val := src.1 c hc
      change (nodes.mem y.region (y.index p c)).read .real=_ at val
      rw [val]
      exact source_graph_lowering inst V x.cols B (values p) c (lt_of_lt_of_le hc cover)
    · intro r i miss
      have off : ¬∃ c,c<y.cols ∧ r=y.region ∧ i=y.index p c := by
        rintro ⟨c,hc,hr,hi⟩
        exact miss p (by simp) c hc ⟨hr.symm,hi.symm⟩
      simp [Floating.access,SoftmaxTriton1.Floating.access,projectState,off]

theorem source_terminates (inst : Instructions) (V : Nat)
    (x y : Tensor) (B p : Nat) (hB : 0<B) (s : State Word) :
    ∃ t,evaluate inst V x y B p s=some t := by
  obtain ⟨nodes,h⟩ := Floating.terminates (algebra inst V) x y B p hB (liftState s)
  exact ⟨projectState V B y p s nodes,by simp [evaluate,h]⟩

#axiomsClean project_active
#axiomsClean source
#axiomsClean source_terminates
end
end VeriTile.Bench.Repairs.TritonSoftmax.NativeGraph
