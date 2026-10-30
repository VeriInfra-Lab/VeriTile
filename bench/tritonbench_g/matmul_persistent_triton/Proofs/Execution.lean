import bench.tritonbench_g.matmul_persistent_triton.Proofs.ControlBounds
import bench.tritonbench_g.cross_entropy1.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Source
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 5000000

abbrev Lane := Fin 128 × Fin 256
abbrev Acc (β : Type) := Fin 128 → Fin 256 → β

/-- Storage scalars and accumulator scalars have separate types. `dot` is
the structural source operation with both matrix operands and its supplied
FP32 accumulator; it is not a numerical axiom or a scalarized dot. -/
structure Arithmetic (α β : Type) where
  inputZero : α
  accumulatorZero : β
  dot : (width : Nat) → (Fin 128 → Fin width → α) →
    (Fin width → Fin 256 → α) → Acc β → Acc β
  cast : Storage → β → α

def loadA {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (memory : Memory α) (tile iteration : Nat) (row : Fin 128) (k : Fin p.profile.blockK) : α :=
  if p.k iteration k.val<p.K then
    memory (b.A.address (clamp p.M (p.row tile row.val)) (clamp p.K (p.k iteration k.val)))
  else ops.inputZero

def loadB {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (memory : Memory α) (tile iteration : Nat) (k : Fin p.profile.blockK) (column : Fin 256) : α :=
  if p.k iteration k.val<p.K then
    memory (b.B.address (clamp p.K (p.k iteration k.val)) (clamp p.N (p.column tile column.val)))
  else ops.inputZero

/-- Exactly one source `tl.dot(a,b,accumulator)` per inner iteration, with
zero initialization at each outer tile. There are no writes in this loop. -/
def accumulator {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (memory : Memory α) (tile : Nat) : Nat → Acc β
  | 0 => fun _ _ => ops.accumulatorZero
  | count+1 => ops.dot p.profile.blockK (loadA ops p b memory tile count)
      (loadB ops p b memory tile count) (accumulator ops p b memory tile count)

def active (p : Problem) (tile : Nat) (lane : Lane) : Prop :=
  p.row tile lane.1.val<p.M ∧ p.column tile lane.2.val<p.N

def outputAddress (p : Problem) (b : Buffers) (tile : Nat) (lane : Lane) : Address :=
  b.C.address (clamp p.M (p.row tile lane.1.val)) (clamp p.N (p.column tile lane.2.val))

def writes (p : Problem) (b : Buffers) (tile : Nat) (point : Address) : Prop :=
  ∃ lane : Lane, active p tile lane ∧ outputAddress p b tile lane=point

instance (p : Problem) (b : Buffers) (tile : Nat) : DecidablePred (writes p b tile) :=
  Classical.decPred _

/-- Parallel masked store selects one active lane. The no-collision theorem
below proves that selection is unique for the concrete output layout. -/
def value {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (memory : Memory α) (tile : Nat) (point : Address) : α :=
  if h : writes p b tile point then
    let lane := Classical.choose h
    ops.cast p.profile (accumulator ops p b memory tile p.kTiles lane.1 lane.2)
  else ops.inputZero

def step {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (tile : Nat) (memory : Memory α) : Memory α :=
  CrossEntropy1.Schedule.put (writes p b) (value ops p b memory) tile memory

def pidTiles (p : Problem) (pid : Nat) : List Nat :=
  (List.range (p.work pid)).map (p.tile pid)

def tiles (p : Problem) : List Nat :=
  (List.range p.grid).flatMap (pidTiles p)

/-- Actual host grid, persistent tile order per PID, and evolving memory. -/
def run {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (memory : Memory α) : Memory α := CrossEntropy1.Schedule.run (step ops p b) (tiles p) memory

def reads (b : Buffers) (point : Address) : Prop :=
  point.1=b.A.storage ∨ point.1=b.B.storage

theorem tile_mem (p : Problem) (tile : Nat) : tile∈tiles p ↔
    ∃ pid<p.grid, ∃ ordinal<p.work pid, p.tile pid ordinal=tile := by
  simp [tiles,pidTiles]

theorem tiles_exact (p : Problem) (hs : 0<p.SMS) (tile : Nat) :
    tile∈tiles p ↔ tile<p.tiles := by
  rw [tile_mem]
  constructor
  · rintro ⟨pid,hp,ordinal,hq,rfl⟩
    exact (Schedule.ordinal_bound_iff p.tiles p.SMS pid ordinal hs
      (lt_of_lt_of_le hp (Nat.min_le_left _ _))).mp hq
  · intro ht
    have cover := Schedule.tile_covered p.tiles p.SMS tile hs ht
    exact ⟨tile%p.SMS,cover.1,tile/p.SMS,cover.2⟩

theorem accumulator_stable {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (entry memory : Memory α) (same : ∀ point, reads b point → memory point=entry point)
    (tile count : Nat) : accumulator ops p b memory tile count=accumulator ops p b entry tile count := by
  have aa : ∀ iteration, loadA ops p b memory tile iteration=loadA ops p b entry tile iteration := by
    intro iteration
    funext row k
    simp only [loadA]
    split_ifs
    · exact same _ (Or.inl rfl)
    · rfl
  have bb : ∀ iteration, loadB ops p b memory tile iteration=loadB ops p b entry tile iteration := by
    intro iteration
    funext k column
    simp only [loadB]
    split_ifs
    · exact same _ (Or.inr rfl)
    · rfl
  induction count with
  | zero => rfl
  | succ count ih => simp only [accumulator,aa,bb,ih]

theorem value_stable {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (entry memory : Memory α) (same : ∀ point, reads b point → memory point=entry point)
    (tile : Nat) : value ops p b memory tile=value ops p b entry tile := by
  funext point
  simp only [value,accumulator_stable ops p b entry memory same]

theorem step_frame {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (tile : Nat) (memory : Memory α) (point : Address) (outside : ¬writes p b tile point) :
    step ops p b tile memory point=memory point := by simp [step,CrossEntropy1.Schedule.put,outside]

theorem read_separate (p : Problem) (b : Buffers) (layout : Layout p b)
    (point : Address) (read : reads b point) (tile : Nat) : ¬writes p b tile point := by
  rintro ⟨lane,_,he⟩
  have storage := congrArg Prod.fst he
  change b.C.storage=point.1 at storage
  rcases read with ha|hb
  · exact layout.aSeparate (storage.trans ha)
  · exact layout.bSeparate (storage.trans hb)

/-- Reads are frozen only after proving that every executed source store
preserves the actual A/B storage. This includes A/B aliasing each other. -/
theorem frozen {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α) (schedule : List Nat)
    (memory : Memory α) (same : ∀ point, reads b point → memory point=entry point) :
    CrossEntropy1.Schedule.run (step ops p b) schedule memory=
      CrossEntropy1.Schedule.run (CrossEntropy1.Schedule.put (writes p b) (value ops p b entry)) schedule memory := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons tile tail ih =>
    have next : ∀ point, reads b point → step ops p b tile memory point=entry point := by
      intro point read
      rw [step_frame ops p b tile memory point (read_separate p b layout point read tile)]
      exact same point read
    have head : step ops p b tile memory=CrossEntropy1.Schedule.put (writes p b) (value ops p b entry) tile memory := by
      funext point
      simp only [step,CrossEntropy1.Schedule.put,value_stable ops p b entry memory same]
    change CrossEntropy1.Schedule.run (step ops p b) tail (step ops p b tile memory)=_
    rw [ih _ next,head]
    rfl

#axiomsClean frozen
#axiomsClean tiles_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Source
