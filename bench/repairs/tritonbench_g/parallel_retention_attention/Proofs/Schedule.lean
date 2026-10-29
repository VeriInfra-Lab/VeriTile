import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Execution

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Interleaved writes from different programs and output regions. -/
def runWrites {ι : Type} (addr : ι → Nat) (value : Memory α → ι → α) : List ι → Memory α → Memory α
  | [],m => m
  | i::is,m => runWrites addr value is (Function.update m (addr i) (value m i))

theorem runWrites_frame {ι : Type} (addr : ι → Nat) (value : Memory α → ι → α)
    (is : List ι) (m : Memory α) (a : Nat) (outside : ∀ i∈is,addr i≠a) :
    runWrites addr value is m a=m a := by
  induction is generalizing m with
  | nil => rfl
  | cons i is ih =>
    simp only [runWrites]
    rw [ih _ (by intro j hj; exact outside j (List.mem_cons_of_mem i hj))]
    exact Function.update_of_ne (Ne.symm (outside i (by simp))) _ _

theorem runWrites_static {ι : Type} (R : Set Nat) (addr : ι → Nat) (value : Memory α → ι → α)
    (locality : ∀ m m',Set.EqOn m m' R → ∀ i,value m i=value m' i)
    (is : List ι) (m initial : Memory α) (same : Set.EqOn m initial R)
    (disjoint : ∀ i∈is,addr i∉R) :
    runWrites addr value is m=runWrites addr (fun _ => value initial) is m := by
  induction is generalizing m with
  | nil => rfl
  | cons i is ih =>
    simp only [runWrites]
    rw [locality m initial same i]
    apply ih
    · intro a ha
      rw [Function.update_of_ne (by intro eq;exact disjoint i (by simp) (eq ▸ ha))]
      exact same ha
    · intro j hj
      exact disjoint j (List.mem_cons_of_mem i hj)

theorem static_write_value {ι : Type} (addr : ι → Nat) (value : ι → α)
    (is : List ι) (m : Memory α) (target : ι) (member : target∈is)
    (unique : ∀ j∈is,addr j=addr target → j=target) :
    runWrites addr (fun _ => value) is m (addr target)=value target := by
  classical
  induction is generalizing m with
  | nil => simp at member
  | cons i is ih =>
    simp only [runWrites]
    by_cases later : target∈is
    · exact ih _ later (by intro j hj;exact unique j (List.mem_cons_of_mem i hj))
    · have eq : target=i := by simpa [later] using member
      subst i
      rw [runWrites_frame addr (fun _ => value) is _ (addr target) (by
        intro j hj eq
        exact later (unique j (List.mem_cons_of_mem target hj) eq ▸ hj))]
      simp

theorem actual_write_value {ι : Type} (R : Set Nat) (addr : ι → Nat) (value : Memory α → ι → α)
    (locality : ∀ m m',Set.EqOn m m' R → ∀ i,value m i=value m' i)
    (is : List ι) (m : Memory α) (disjoint : ∀ i∈is,addr i∉R)
    (target : ι) (member : target∈is) (unique : ∀ j∈is,addr j=addr target → j=target) :
    runWrites addr value is m (addr target)=value m target := by
  rw [runWrites_static R addr value locality is m m (by intro a ha;rfl) disjoint]
  exact static_write_value addr (value m) is m target member unique

structure Cell where
  kind : OutputKind
  index : Nat
  deriving DecidableEq

def allCells (s : Shape) (scratch : Bool) : List Cell :=
  [.query,.key,.value].flatMap fun kind =>
    (List.range (if scratch then kind.count s else s.B*s.H*s.T*kind.width s)).map (Cell.mk kind)

def cellsAddress (bases : OutputKind → Nat) (cell : Cell) := bases cell.kind+cell.index

def backwardLaunch (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Cell) (m : Memory α) :=
  runWrites (cellsAddress bases) (fun mem cell => scratchCurrent a p s b cell.kind mem cell.index) schedule m

theorem backward_launch_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Cell) (m : Memory α)
    (fresh : ∀ cell∈schedule,cellsAddress bases cell∉inputCells s b)
    (target : Cell) (member : target∈schedule)
    (unique : ∀ cell∈schedule,cellsAddress bases cell=cellsAddress bases target → cell=target) :
    backwardLaunch a p s b bases schedule m (cellsAddress bases target)=
      scratchValue a p s target.kind (currentInputs a s b m) target.index := by
  apply actual_write_value (inputCells s b) (cellsAddress bases)
    (fun mem cell => scratchCurrent a p s b cell.kind mem cell.index) _ schedule m fresh target member unique
  intro mem mem' same cell
  exact scratch_local a p s b cell.kind mem mem' same cell.index

/-- The source's barriers and per-program DQ/DKV order may choose any
serialization of these independent stores; all exterior cells are retained. -/
theorem backward_launch_frame (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Cell) (m : Memory α)
    (address : Nat) (outside : ∀ cell∈schedule,cellsAddress bases cell≠address) :
    backwardLaunch a p s b bases schedule m address=m address :=
  runWrites_frame _ _ _ _ _ outside

#axiomsClean runWrites_frame
#axiomsClean runWrites_static
#axiomsClean actual_write_value
#axiomsClean backward_launch_output
#axiomsClean backward_launch_frame
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
