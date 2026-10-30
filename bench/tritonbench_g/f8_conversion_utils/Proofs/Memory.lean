import bench.tritonbench_g.f8_conversion_utils.Proofs.Chunks

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Memory
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 4000000

/-- Different input and output carriers prevent assuming a lossless cast.
Only ordinary local load/store laws occur here. Real may use arbitrary Real
carriers; native instantiations use words of exactly their storage width. -/
structure Primitives (M A B : Type) (inputWidth outputWidth : Nat) where
  readInput : M → Nat → A
  readOutput : M → Nat → B
  writeOutput : M → Nat → B → M
  byte : M → Nat → Fin 256
  read_same : ∀ s a value,readOutput (writeOutput s a value) a=value
  read_disjoint : ∀ s a value b,
    SpanDisjoint a (a+outputWidth) b (b+outputWidth) →
    readOutput (writeOutput s a value) b=readOutput s b
  input_disjoint : ∀ s a value b,
    SpanDisjoint a (a+outputWidth) b (b+inputWidth) →
    readInput (writeOutput s a value) b=readInput s b
  byte_frame : ∀ s a value b,b<a ∨ a+outputWidth≤b →
    byte (writeOutput s a value) b=byte s b

variable {M A B : Type} {iw ow : Nat} (ops : Primitives M A B iw ow)

def address (base width i : Nat) : Nat := base+i*width

def stores (base : Nat) (values : Nat → B) : List Nat → M → M
  | [],s => s
  | i::rest,s => stores base values rest (ops.writeOutput s (address base ow i) (values i))

theorem stores_append (base : Nat) (values : Nat → B) (xs ys : List Nat) (s : M) :
    stores ops base values (xs++ys) s=stores ops base values ys (stores ops base values xs s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => exact ih _

theorem distinct (base width i j : Nat) (different : i≠j) :
    SpanDisjoint (address base width i) (address base width i+width)
      (address base width j) (address base width j+width) := by
  unfold SpanDisjoint address
  rcases Nat.lt_or_gt_of_ne different with h|h
  · left
    have := Nat.mul_le_mul_right width (show i+1≤j by omega)
    simpa only [Nat.add_mul,Nat.one_mul,Nat.add_assoc] using Nat.add_le_add_left this base
  · right
    have := Nat.mul_le_mul_right width (show j+1 ≤ i by omega)
    simpa only [Nat.add_mul,Nat.one_mul,Nat.add_assoc] using Nat.add_le_add_left this base

theorem stores_value (base : Nat) (values : Nat → B) (jobs : List Nat) (s : M) (i : Nat) :
    ops.readOutput (stores ops base values jobs s) (address base ow i)=
      if i∈jobs then values i else ops.readOutput s (address base ow i) := by
  induction jobs generalizing s with
  | nil => simp [stores]
  | cons j rest ih =>
    rw [stores,ih]
    by_cases member : i∈rest
    · simp [member]
    · by_cases same : j=i
      · subst j
        simp [member,ops.read_same]
      · rw [ops.read_disjoint _ _ _ _ (distinct base ow j i same)]
        simp [member,Ne.symm same]

theorem stores_input (base input : Nat) (values : Nat → B) (jobs : List Nat) (s : M)
    (apart : ∀ j∈jobs,SpanDisjoint (address base ow j) (address base ow j+ow) input (input+iw)) :
    ops.readInput (stores ops base values jobs s) input=ops.readInput s input := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,ih _ (fun k hk => apart k (by simp [hk]))]
    exact ops.input_disjoint _ _ _ _ (apart j (by simp))

theorem stores_frame (base : Nat) (values : Nat → B) (jobs : List Nat) (s : M) (q : Nat)
    (outside : ∀ j∈jobs,q<address base ow j ∨ address base ow j+ow≤q) :
    ops.byte (stores ops base values jobs s) q=ops.byte s q := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [stores,ih _ (fun k hk => outside k (by simp [hk]))]
    exact ops.byte_frame _ _ _ _ (outside j (by simp))

def writes (twice : Bool) (jobs : List Nat) : List Nat := if twice then jobs++jobs else jobs

@[simp] theorem mem_writes (twice : Bool) (jobs : List Nat) (i : Nat) :
    i∈writes twice jobs ↔ i∈jobs := by cases twice <;> simp [writes]

/-- Actual loaded input registers are retained across both decode stores.
No cast-idempotence law or store-address uniqueness is required. -/
def run (twice : Bool) (compute : A → B) (input output : Nat) : List (List Nat) → M → M
  | [],s => s
  | jobs::rest,s =>
    let values := fun i => compute (ops.readInput s (address input iw i))
    run twice compute input output rest (stores ops output values (writes twice jobs) s)

def writeSchedule (twice : Bool) (groups : List (List Nat)) : List Nat :=
  groups.flatMap (writes twice)

@[simp] theorem mem_schedule (twice : Bool) (groups : List (List Nat)) (i : Nat) :
    i∈writeSchedule twice groups ↔ i∈groups.flatten := by
  simp only [writeSchedule,List.mem_flatMap,mem_writes,List.mem_flatten]

theorem buffered (twice : Bool) (compute : A → B) (input output n : Nat)
    (groups : List (List Nat)) (original current : M)
    (active : ∀ i∈groups.flatten,i<n)
    (apart : ∀ i,i<n → ∀ j,j<n →
      SpanDisjoint (address output ow i) (address output ow i+ow)
        (address input iw j) (address input iw j+iw))
    (preserved : ∀ i,i<n → ops.readInput current (address input iw i)=
      ops.readInput original (address input iw i)) :
    run ops twice compute input output groups current=
      stores ops output (fun i => compute (ops.readInput original (address input iw i)))
        (writeSchedule twice groups) current := by
  induction groups generalizing current with
  | nil => rfl
  | cons jobs rest ih =>
    have hj (i : Nat) (hi : i∈writes twice jobs) : i<n :=
      active i (by simp only [List.flatten_cons,List.mem_append];exact Or.inl ((mem_writes twice jobs i).mp hi))
    have values : stores ops output (fun i => compute (ops.readInput current (address input iw i)))
        (writes twice jobs) current=
        stores ops output (fun i => compute (ops.readInput original (address input iw i)))
          (writes twice jobs) current := by
      have congrStores (xs : List Nat) (ha : ∀ i∈xs,i<n) (s : M) :
          stores ops output (fun i => compute (ops.readInput current (address input iw i))) xs s=
          stores ops output (fun i => compute (ops.readInput original (address input iw i))) xs s := by
        induction xs generalizing s with
        | nil => rfl
        | cons i xs ih =>
          simp only [stores,preserved i (ha i (by simp))]
          exact ih (fun j h => ha j (by simp [h])) _
      exact congrStores _ hj _
    rw [run,values,ih]
    · simp only [writeSchedule,List.flatMap_cons,stores_append]
    · intro i hi
      exact active i (by simp [hi])
    · intro j hj'
      rw [stores_input ops output (address input iw j) _ _ _ (fun i hi => apart i (hj i hi) j hj')]
      exact preserved j hj'

theorem complete (twice : Bool) (compute : A → B) (input output n : Nat)
    (groups : List (List Nat)) (original : M)
    (coverage : ∀ i,i∈groups.flatten ↔ i<n)
    (apart : ∀ i,i<n → ∀ j,j<n →
      SpanDisjoint (address output ow i) (address output ow i+ow)
        (address input iw j) (address input iw j+iw)) :
    (∀ i,i<n → ops.readOutput (run ops twice compute input output groups original)
      (address output ow i)=compute (ops.readInput original (address input iw i))) ∧
    (∀ q,(∀ i,i<n → q<address output ow i ∨ address output ow i+ow≤q) →
      ops.byte (run ops twice compute input output groups original) q=ops.byte original q) := by
  rw [buffered ops twice compute input output n groups original original
    (fun i hi => (coverage i).mp hi) apart (fun _ _ => rfl)]
  constructor
  · intro i hi
    rw [stores_value,if_pos ((mem_schedule twice groups i).mpr ((coverage i).mpr hi))]
  · intro q outside
    apply stores_frame
    intro i hi
    exact outside i ((coverage i).mp ((mem_schedule twice groups i).mp hi))

#axiomsClean buffered
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Memory
