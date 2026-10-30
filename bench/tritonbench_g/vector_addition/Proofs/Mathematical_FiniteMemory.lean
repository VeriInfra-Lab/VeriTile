import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteViews

namespace VeriTile.Bench.TritonBenchG.VectorAddition.Finite
open DTypeScope
set_option maxHeartbeats 3000000

/-- Trusted ordinary typed load/store primitives and their local memory laws.
No operation computes the kernel value or supplies execution success. Byte
observations make the exterior memory frame explicit without a second byte
interpreter for every scalar dtype. -/
structure MemoryPrimitives (M A : Type) where
  launchDevice : Nat
  readableByte : Nat → Prop
  writableByte : Nat → Prop
  read : M → DType → Nat → A
  write : M → DType → Nat → A → M
  observeByte : M → Nat → Fin 256
  read_same : ∀ s d a v, read (write s d a v) d a=v
  read_disjoint : ∀ s d a v e b,
    SpanDisjoint a (a+bytes d) b (b+bytes e) → read (write s d a v) e b=read s e b
  byte_frame : ∀ s d a v b,
    b<a ∨ a+bytes d≤b → observeByte (write s d a v) b=observeByte s b

/-- Allocation/permission applicability for this concrete launch address space.
The trusted primitive functions may have total mathematical extensions, but the
source theorem only uses loads/stores on these explicitly permitted bytes.
Permissions are the stable allocation permissions for the duration of launch. -/
structure MemoryApplicable {M A : Type} (ops : MemoryPrimitives M A)
    (x y output : View) (n : Nat) : Prop where
  deviceX : ops.launchDevice=x.device
  deviceY : ops.launchDevice=y.device
  deviceOutput : ops.launchDevice=output.device
  readX : ∀ i byte, i<n → byte<bytes x.dtype → ops.readableByte (x.address i+byte)
  readY : ∀ i byte, i<n → byte<bytes y.dtype → ops.readableByte (y.address i+byte)
  readOutput : ∀ i byte, i<n → byte<bytes output.dtype → ops.readableByte (output.address i+byte)
  writeOutput : ∀ i byte, i<n → byte<bytes output.dtype → ops.writableByte (output.address i+byte)

def readView {M A : Type} (ops : MemoryPrimitives M A) (s : M) (v : View) (i : Nat) : A :=
  ops.read s v.dtype (v.address i)

def writeView {M A : Type} (ops : MemoryPrimitives M A) (s : M) (v : View) (i : Nat) (a : A) : M :=
  ops.write s v.dtype (v.address i) a

/-- Active source memory operations: each lane loads x and y from current
memory, applies its concrete typed scalar computation, and stores output. -/
def runLive {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (x y output : View) : List Nat → M → M
  | [],s => s
  | i::rest,s =>
      let value := compute (readView ops s x i) (readView ops s y i)
      runLive ops compute x y output rest (writeView ops s output i value)

/-- Buffered loads with actual sequential stores; used only after proving that
output writes cannot affect either input. -/
def runLoaded {M A : Type} (ops : MemoryPrimitives M A) (values : Nat → A)
    (output : View) : List Nat → M → M
  | [],s => s
  | i::rest,s => runLoaded ops values output rest (writeView ops s output i (values i))

def InputsPreserved {M A : Type} (ops : MemoryPrimitives M A)
    (x y : View) (n : Nat) (base current : M) : Prop :=
  ∀ i, i<n → readView ops current x i=readView ops base x i ∧
    readView ops current y i=readView ops base y i

theorem write_preserves_inputs {M A : Type} (ops : MemoryPrimitives M A)
    (x y output : View) (n i : Nat) (base current : M) (value : A)
    (preserved : InputsPreserved ops x y n base current)
    (hx : ∀ j, j<n → SpanDisjoint (output.address i) (output.finish i) (x.address j) (x.finish j))
    (hy : ∀ j, j<n → SpanDisjoint (output.address i) (output.finish i) (y.address j) (y.finish j)) :
    InputsPreserved ops x y n base (writeView ops current output i value) := by
  intro j hj
  constructor
  · exact (ops.read_disjoint current output.dtype (output.address i) value x.dtype (x.address j) (hx j hj)).trans (preserved j hj).1
  · exact (ops.read_disjoint current output.dtype (output.address i) value y.dtype (y.address j) (hy j hj)).trans (preserved j hj).2

theorem live_eq_loaded_aux {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (x y output : View) (n : Nat) (base : M) (jobs : List Nat)
    (active : ∀ i∈jobs, i<n)
    (hx : ∀ i, i<n → ∀ j, j<n → SpanDisjoint (output.address i) (output.finish i) (x.address j) (x.finish j))
    (hy : ∀ i, i<n → ∀ j, j<n → SpanDisjoint (output.address i) (output.finish i) (y.address j) (y.finish j))
    (current : M) (preserved : InputsPreserved ops x y n base current) :
    runLive ops compute x y output jobs current=
    runLoaded ops (fun i ↦ compute (readView ops base x i) (readView ops base y i)) output jobs current := by
  induction jobs generalizing current with
  | nil => rfl
  | cons i jobs ih =>
    have hi := active i (by simp)
    have hp := preserved i hi
    simp only [runLive,runLoaded,hp.1,hp.2]
    apply ih (fun i h ↦ active i (by simp [h]))
    exact write_preserves_inputs ops x y output n i base current _ preserved (hx i hi) (hy i hi)

theorem source_load_all_store_correspondence {M A : Type} (ops : MemoryPrimitives M A)
    (compute : A → A → A) (x y output : View) (n : Nat) (base : M)
    (jobs : List Nat) (active : ∀ i∈jobs, i<n)
    (hx : x.Valid n (rounded n)) (hy : y.Valid n (rounded n)) (ho : output.Valid n (rounded n))
    (dx : StorageDisjoint output x) (dy : StorageDisjoint output y) :
    runLive ops compute x y output jobs base=
    runLoaded ops (fun i ↦ compute (readView ops base x i) (readView ops base y i)) output jobs base := by
  apply live_eq_loaded_aux ops compute x y output n base jobs active
  · intro i hi j hj
    exact separated_accesses output x n n (rounded n) (rounded n) i j ho hx hi hj dx
  · intro i hi j hj
    exact separated_accesses output y n n (rounded n) (rounded n) i j ho hy hi hj dy
  · intro i _
    exact ⟨rfl,rfl⟩

theorem loaded_output {M A : Type} (ops : MemoryPrimitives M A) (values : Nat → A)
    (output : View) (jobs : List Nat) (initial : M) (i : Nat) :
    readView ops (runLoaded ops values output jobs initial) output i=
      if i∈jobs then values i else readView ops initial output i := by
  induction jobs generalizing initial with
  | nil => simp [runLoaded]
  | cons j jobs ih =>
    simp only [runLoaded,ih]
    by_cases hjobs : i∈jobs
    · simp [hjobs]
    · by_cases hji : j=i
      · subst j
        simp [hjobs,readView,writeView,ops.read_same]
      · have hd := distinct_cells output j i hji
        have hread := ops.read_disjoint initial output.dtype (output.address j) (values j)
          output.dtype (output.address i) hd
        simp [hjobs,Ne.symm hji,readView,writeView,hread]

theorem loaded_byte_frame {M A : Type} (ops : MemoryPrimitives M A) (values : Nat → A)
    (output : View) (jobs : List Nat) (initial : M) (address : Nat)
    (outside : ∀ i∈jobs, address<output.address i ∨ output.finish i≤address) :
    ops.observeByte (runLoaded ops values output jobs initial) address=ops.observeByte initial address := by
  induction jobs generalizing initial with
  | nil => rfl
  | cons i jobs ih =>
    rw [runLoaded,ih _ (fun j hj ↦ outside j (by simp [hj]))]
    exact ops.byte_frame initial output.dtype (output.address i) (values i) address (outside i (by simp))

/-- Exact source program IDs/lanes with the original active mask. -/
def activeLanes (n : Nat) : List Nat :=
  (List.range ((n+1023)/1024)).flatMap fun pid ↦
    ((List.range 1024).map (fun lane ↦ pid*1024+lane)).filter (fun i ↦ decide (i<n))

theorem mem_activeLanes (n i : Nat) : i∈activeLanes n ↔ i<n := by
  constructor
  · intro h
    simp only [activeLanes,List.mem_flatMap,List.mem_filter,decide_eq_true_eq] at h
    obtain ⟨pid,_,_,hi⟩ := h
    exact hi
  · intro hi
    obtain ⟨hp,hl,he⟩ := active_owner n i hi
    simp only [activeLanes,List.mem_flatMap,List.mem_filter,List.mem_map,List.mem_range,decide_eq_true_eq]
    exact ⟨i/1024,hp,⟨i%1024,hl,he⟩,hi⟩

#axiomsClean write_preserves_inputs
#axiomsClean live_eq_loaded_aux
#axiomsClean source_load_all_store_correspondence
#axiomsClean loaded_output
#axiomsClean loaded_byte_frame
#axiomsClean mem_activeLanes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.Finite
