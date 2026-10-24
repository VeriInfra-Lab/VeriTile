import bench.tritonbench_g.vector_addition.Proofs.FiniteRoutes
import bench.tritonbench_g.vector_addition.Proofs.FiniteSchedule
import bench.repairs.tritonbench_g.dropout_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.DropoutTriton.Physical
open VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Triton DTypeScope TypedHost Finite
noncomputable section
open Classical
set_option maxHeartbeats 4000000

theorem full_coverage (n i : Nat) : i ∈ (sourceBatches n).flatten ↔ i < n := by
  rw [source_batches_flatten]
  exact mem_activeLanes n i

/-- Heterogeneous local primitives. Data stores are exactly four bytes;
the keep load uses its own dtype and width, with no reinterpretation as FP32.
`K` can be the dependent raw keep value or its independently checked truth.
Only disjointness and ordinary read-after-write laws occur here. -/
structure Memory (M A K : Type) where
  read : M → Nat → A
  readKeep : M → DType → Nat → K
  write : M → Nat → A → M
  byte : M → Nat → Fin 256
  read_same : ∀ s a v, read (write s a v) a=v
  read_disjoint : ∀ s a v b, SpanDisjoint a (a+4) b (b+4) →
    read (write s a v) b=read s b
  keep_disjoint : ∀ s a v d b, SpanDisjoint a (a+4) b (b+bytes d) →
    readKeep (write s a v) d b=readKeep s d b
  byte_frame : ∀ s a v b, b<a ∨ a+4≤b → byte (write s a v) b=byte s b

def readX (ops : Memory M A K) (s : M) (v : View) (i : Nat) : A := ops.read s (v.address i)
def readK (ops : Memory M A K) (s : M) (v : View) (i : Nat) : K := ops.readKeep s v.dtype (v.address i)
def writeO (ops : Memory M A K) (s : M) (v : View) (i : Nat) (a : A) : M := ops.write s (v.address i) a

def runLoaded (ops : Memory M A K) (values : Nat → A) (o : View) : List Nat → M → M
  | [],s => s
  | i::rest,s => runLoaded ops values o rest (writeO ops s o i (values i))

def runLive (ops : Memory M A K) (compute : A → K → A) (x k o : View) : List Nat → M → M
  | [],s => s
  | i::rest,s => runLive ops compute x k o rest
      (writeO ops s o i (compute (readX ops s x i) (readK ops s k i)))

/-- A program loads both complete active tiles before any store. Inactive
register arithmetic is tracked separately by the source interpreter. -/
def runBatches (ops : Memory M A K) (compute : A → K → A) (x k o : View) : List (List Nat) → M → M
  | [],s => s
  | jobs::rest,s => runBatches ops compute x k o rest
      (runLoaded ops (fun i => compute (readX ops s x i) (readK ops s k i)) o jobs s)

def runSource (ops : Memory M A K) (compute : A → K → A) (x k o : View) (n : Nat) (s : M) : M :=
  runBatches ops compute x k o (sourceBatches n) s

theorem live_append (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (as bs : List Nat) (s : M) :
    runLive ops compute x k o (as++bs) s=runLive ops compute x k o bs (runLive ops compute x k o as s) := by
  induction as generalizing s with
  | nil => rfl
  | cons i rest ih => exact ih _

theorem live_eq_loaded_aux (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (n : Nat) (base : M) (jobs : List Nat) (hx : x.dtype=.fp32) (ho : o.dtype=.fp32)
    (active : ∀ i∈jobs,i<n)
    (dx : ∀ i, i<n → ∀ j,j<n → SpanDisjoint (o.address i) (o.finish i) (x.address j) (x.finish j))
    (dk : ∀ i, i<n → ∀ j,j<n → SpanDisjoint (o.address i) (o.finish i) (k.address j) (k.finish j))
    (current : M)
    (preserved : ∀ i,i<n → readX ops current x i=readX ops base x i ∧
      readK ops current k i=readK ops base k i) :
    runLive ops compute x k o jobs current=
      runLoaded ops (fun i => compute (readX ops base x i) (readK ops base k i)) o jobs current := by
  induction jobs generalizing current with
  | nil => rfl
  | cons i rest ih =>
    have hi := active i (by simp)
    have hp := preserved i hi
    simp only [runLive,runLoaded,hp.1,hp.2]
    apply ih (fun j hj => active j (by simp [hj]))
    intro j hj
    constructor
    · apply Eq.trans (ops.read_disjoint current (o.address i) _ (x.address j) ?_) (preserved j hj).1
      simpa [View.finish,ho,hx,bytes] using dx i hi j hj
    · apply Eq.trans (ops.keep_disjoint current (o.address i) _ k.dtype (k.address j) ?_) (preserved j hj).2
      simpa [View.finish,ho,bytes] using dk i hi j hj

theorem live_loaded (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (n : Nat) (s : M) (jobs : List Nat) (active : ∀ i∈jobs,i<n)
    (hx : x.dtype=.fp32) (ho : o.dtype=.fp32)
    (vx : x.Valid n n) (vk : k.Valid n n) (vo : o.Valid n n)
    (dx : StorageDisjoint o x) (dk : StorageDisjoint o k) :
    runLive ops compute x k o jobs s=
      runLoaded ops (fun i => compute (readX ops s x i) (readK ops s k i)) o jobs s := by
  apply live_eq_loaded_aux ops compute x k o n s jobs hx ho active
  · exact fun i hi j hj => separated_accesses o x n n n n i j vo vx hi hj dx
  · exact fun i hi j hj => separated_accesses o k n n n n i j vo vk hi hj dk
  · exact fun _ _ => ⟨rfl,rfl⟩

theorem batches_live (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (n : Nat) (lists : List (List Nat)) (s : M) (active : ∀ i∈lists.flatten,i<n)
    (hx : x.dtype=.fp32) (ho : o.dtype=.fp32)
    (vx : x.Valid n n) (vk : k.Valid n n) (vo : o.Valid n n)
    (dx : StorageDisjoint o x) (dk : StorageDisjoint o k) :
    runBatches ops compute x k o lists s=runLive ops compute x k o lists.flatten s := by
  induction lists generalizing s with
  | nil => rfl
  | cons jobs rest ih =>
    have ha : ∀ i∈jobs,i<n := fun i hi => active i (by simp [hi])
    have hr : ∀ i∈rest.flatten,i<n := fun i hi => active i (by simp [hi])
    rw [runBatches,ih _ hr,List.flatten_cons,live_append,
      live_loaded ops compute x k o n s jobs ha hx ho vx vk vo dx dk]

theorem loaded_output (ops : Memory M A K) (values : Nat → A) (o : View)
    (ho : o.dtype=.fp32) (jobs : List Nat) (s : M) (i : Nat) :
    readX ops (runLoaded ops values o jobs s) o i=
      if i∈jobs then values i else readX ops s o i := by
  induction jobs generalizing s with
  | nil => simp [runLoaded]
  | cons j rest ih =>
    simp only [runLoaded,ih]
    by_cases hr : i∈rest
    · simp [hr]
    · by_cases hji : j=i
      · subst j; simp [hr,readX,writeO,ops.read_same]
      · have hd : SpanDisjoint (o.address j) (o.address j+4) (o.address i) (o.address i+4) := by
          simpa [View.finish,ho,bytes] using distinct_cells o j i hji
        have he := ops.read_disjoint s (o.address j) (values j) (o.address i) hd
        simp [hr,Ne.symm hji,readX,writeO,he]

theorem live_byte_frame (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (n : Nat) (ho : o.dtype=.fp32) (jobs : List Nat) (s : M)
    (active : ∀ i∈jobs,i<n) (address : Nat) (outside : ¬o.active n address) :
    ops.byte (runLive ops compute x k o jobs s) address=ops.byte s address := by
  induction jobs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [runLive,ih _ (fun j hj => active j (by simp [hj]))]
    apply ops.byte_frame
    simpa [View.finish,ho,bytes] using outside_span o n i address (active i (by simp)) outside

/-- Full single-launch grid/tail values and every backing byte outside output, with
heterogeneous keep loads and no total-size bound or numerical axiom. -/
theorem source_complete (ops : Memory M A K) (compute : A → K → A) (x k o : View)
    (n : Nat) (s : M) (hx : x.dtype=.fp32) (ho : o.dtype=.fp32)
    (vx : x.Valid n n) (vk : k.Valid n n) (vo : o.Valid n n)
    (dx : StorageDisjoint o x) (dk : StorageDisjoint o k) :
    (∀ i,i<n → readX ops (runSource ops compute x k o n s) o i=
      compute (readX ops s x i) (readK ops s k i)) ∧
    (∀ a,¬o.active n a → ops.byte (runSource ops compute x k o n s) a=ops.byte s a) := by
  have active : ∀ i∈(sourceBatches n).flatten,i<n := fun i hi => (full_coverage n i).mp hi
  have he := batches_live ops compute x k o n (sourceBatches n) s active hx ho vx vk vo dx dk
  constructor
  · intro i hi
    rw [runSource,he,live_loaded ops compute x k o n s _ active hx ho vx vk vo dx dk,loaded_output]
    simp [(full_coverage n i).mpr hi]
    exact ho
  · intro a ha
    rw [runSource,he]
    exact live_byte_frame ops compute x k o n ho _ s active a ha

#axiomsClean source_complete
#axiomsClean live_eq_loaded_aux
#axiomsClean loaded_output
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Physical
