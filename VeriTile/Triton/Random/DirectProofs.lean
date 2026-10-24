import VeriTile.Triton.Random.Direct

/-! Correspondence proofs for the unchanged direct Philox shape semantics. -/
namespace VeriTile.Triton.Random.Direct

set_option maxRecDepth 10000

theorem run_step {s out : Shapes} {a b : TileShape} {n : Nat}
    (bc21 : Broadcast s.c2 s.c1 a) (bc03 : Broadcast s.c0 s.c3 b)
    (rest : Plan ⟨a, s.c2, b, s.c0⟩ n out) (v : Words s) (key : Key) :
    run (.step bc21 bc03 rest) v key =
      run rest (round bc21 bc03 v key) (bumpKey key) := by
  have h : run (.step bc21 bc03 rest) =
      fun v key => run rest (round bc21 bc03 v key) (bumpKey key) := rfl
  exact congrFun (congrFun h v) key

theorem philox_succ (n : Nat) (counter : Counter) (key : Key) :
    philox (n+1) counter key = philox n (Random.step counter key) (bumpKey key) := by
  have h : philox (n+1) =
      fun counter key => philox n (Random.step counter key) (bumpKey key) := rfl
  exact congrFun (congrFun h counter) key

@[simp] theorem zero_rounds {s : Shapes} (v : Words s) (key : Key) :
    run (.done : Plan s 0 s) v key = v := rfl

def sameShapes (shape : TileShape) : Shapes := ⟨shape, shape, shape, shape⟩

def selfBroadcast : (shape : TileShape) → Broadcast shape shape shape
  | [] => .nil
  | _ :: rest => .consSame (selfBroadcast rest)

@[simp] theorem self_left (shape : TileShape) (i : TileIndex shape) :
    (selfBroadcast shape).leftIndex i = i := by
  induction shape with
  | nil => cases i; rfl
  | cons n rest ih =>
      rcases i with ⟨i, tail⟩
      simp [selfBroadcast, ih]

@[simp] theorem self_right (shape : TileShape) (i : TileIndex shape) :
    (selfBroadcast shape).rightIndex i = i := by
  induction shape with
  | nil => cases i; rfl
  | cons n rest ih =>
      rcases i with ⟨i, tail⟩
      simp [selfBroadcast, ih]

def scalarRight : (shape : TileShape) → Broadcast shape [] shape
  | [] => .nil
  | _ :: _ => .scalarR

@[simp] theorem scalarRight_left (shape : TileShape) (i : TileIndex shape) :
    (scalarRight shape).leftIndex i = i := by
  cases shape with
  | nil => cases i; rfl
  | cons => rfl

@[simp] theorem scalarRight_right (shape : TileShape) (i : TileIndex shape) :
    (scalarRight shape).rightIndex i = PUnit.unit := by cases shape <;> rfl

def ofPointwise {shape : TileShape} (f : TileIndex shape → Counter) : Words (sameShapes shape) :=
  { c0 := fun i => (f i).c0, c1 := fun i => (f i).c1
    c2 := fun i => (f i).c2, c3 := fun i => (f i).c3 }

def pointwise {shape : TileShape} (v : Words (sameShapes shape)) (i : TileIndex shape) : Counter :=
  ⟨v.c0 i, v.c1 i, v.c2 i, v.c3 i⟩

@[simp] theorem pointwise_ofPointwise {shape : TileShape}
    (f : TileIndex shape → Counter) (i : TileIndex shape) :
    pointwise (ofPointwise f) i = f i := by rfl

theorem round_pointwise {shape : TileShape} (f : TileIndex shape → Counter) (key : Key) :
    round (selfBroadcast shape) (selfBroadcast shape) (ofPointwise f) key =
      ofPointwise (fun i => Random.step (f i) key) := by
  apply Words.ext <;> funext i
  all_goals dsimp only [round, ofPointwise, Random.step, sameShapes]
  all_goals first | rfl | simp only [self_left, self_right]

def samePlan (shape : TileShape) : (n : Nat) → Plan (sameShapes shape) n (sameShapes shape)
  | 0 => .done
  | n+1 => .step (selfBroadcast shape) (selfBroadcast shape) (samePlan shape n)

theorem samePlan_succ (shape : TileShape) (n : Nat) :
    samePlan shape (n+1) =
      .step (selfBroadcast shape) (selfBroadcast shape) (samePlan shape n) := rfl

theorem samePlan_pointwise {shape : TileShape} (n : Nat)
    (f : TileIndex shape → Counter) (key : Key) :
    run (samePlan shape n) (ofPointwise f) key =
      ofPointwise (fun i => philox n (f i) key) := by
  induction n generalizing f key with
  | zero => rfl
  | succ n ih =>
      rw [samePlan_succ, run_step, round_pointwise]
      exact (ih (fun i => Random.step (f i) key) (bumpKey key)).trans
        (congrArg ofPointwise (funext fun i => (philox_succ n (f i) key).symm))

/-- The actual uniform source has a scalar high counter and three tiles.
Its first round broadcasts the high counter; subsequent rounds keep the tile
shape. The zero-round source profile still has a scalar second result. -/
def uniformInputShapes (shape : TileShape) : Shapes := ⟨shape, [], shape, shape⟩

def uniformInput {shape : TileShape} (c0 c2 c3 : TileIndex shape → Word) (c1 : Word) :
    Words (uniformInputShapes shape) :=
  ⟨c0, fun _ => c1, c2, c3⟩

def uniformPlan (shape : TileShape) (remaining : Nat) :
    Plan (uniformInputShapes shape) (remaining+1) (sameShapes shape) :=
  .step (scalarRight shape) (selfBroadcast shape) (samePlan shape remaining)

theorem uniform_first_round {shape : TileShape}
    (c0 c2 c3 : TileIndex shape → Word) (c1 : Word) (key : Key) :
    round (scalarRight shape) (selfBroadcast shape) (uniformInput c0 c2 c3 c1) key =
      ofPointwise (fun i => Random.step ⟨c0 i, c1, c2 i, c3 i⟩ key) := by
  apply Words.ext <;> funext i
  all_goals dsimp only [round, uniformInput, ofPointwise, Random.step,
    sameShapes, uniformInputShapes]
  all_goals first | rfl | simp only [scalarRight_left, self_left, self_right]

theorem uniformPlan_pointwise {shape : TileShape} (remaining : Nat)
    (c0 c2 c3 : TileIndex shape → Word) (c1 : Word) (key : Key) :
    run (uniformPlan shape remaining) (uniformInput c0 c2 c3 c1) key =
      ofPointwise (fun i => philox (remaining+1) ⟨c0 i, c1, c2 i, c3 i⟩ key) := by
  rw [uniformPlan, run_step, uniform_first_round]
  exact (samePlan_pointwise remaining
    (fun i => Random.step ⟨c0 i, c1, c2 i, c3 i⟩ key) (bumpKey key)).trans
    (congrArg ofPointwise
      (funext fun i => (philox_succ remaining ⟨c0 i, c1, c2 i, c3 i⟩ key).symm))

end VeriTile.Triton.Random.Direct
