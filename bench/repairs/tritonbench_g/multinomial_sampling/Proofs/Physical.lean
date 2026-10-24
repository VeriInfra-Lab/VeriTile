import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Host

namespace VeriTile.Bench.Repairs.MultinomialSampling.Physical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

/-- Ordinary local 64-bit store/read and byte-frame contracts. No field talks
about kernel outputs, categorical results, launches, or whole memory equality. -/
structure API (α Store Byte : Type) where
  readScore : Store → Nat → α
  read : Store → Nat → BitVec 64
  write : Store → Nat → BitVec 64 → Store
  byte : Store → Nat → Byte
  read_same : ∀ m a v, read (write m a v) a = v
  read_apart : ∀ m a b v, (a+8 ≤ b ∨ b+8 ≤ a) → read (write m a v) b = read m b
  score_apart : ∀ m a b v, (a+8 ≤ b ∨ b+4 ≤ a) → readScore (write m a v) b = readScore m b
  byte_frame : ∀ m a b v, (b < a ∨ a+8 ≤ b) → byte (write m a v) b = byte m b

inductive DType where
  | fp32 | int64
  deriving DecidableEq

structure Tensor where
  handle : Nat
  device : Nat
  base : Nat
  allocationEnd : Nat
  shape : List Nat
  strides : List Nat
  elementBytes : Nat
  dtype : DType

/-- Row read through the actual returned Tensor descriptor. The singleton
shape has a second coordinate fixed to zero. -/
def logicalRead (api : API α Store Byte) (m : Store) (tensor : Tensor) (row : Nat) : BitVec 64 :=
  api.read m (tensor.base+8*(row*tensor.strides.headD 1))

structure Write where
  address : Nat
  value : BitVec 64

def runWrites (api : API α Store Byte) (events : List Write) (m : Store) : Store :=
  events.foldl (fun state e => api.write state e.address e.value) m

theorem writes_read_frame (api : API α Store Byte) (events : List Write)
    (m : Store) (a : Nat) (h : ∀ e ∈ events, e.address+8 ≤ a ∨ a+8 ≤ e.address) :
    api.read (runWrites api events m) a = api.read m a := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.read (runWrites api es (api.write m e.address e.value)) a = _
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.read_apart m _ _ _ (h e (List.mem_cons_self ..))

theorem writes_score_frame (api : API α Store Byte) (events : List Write)
    (m : Store) (a : Nat) (h : ∀ e ∈ events, e.address+8 ≤ a ∨ a+4 ≤ e.address) :
    api.readScore (runWrites api events m) a = api.readScore m a := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.readScore (runWrites api es (api.write m e.address e.value)) a = _
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.score_apart m _ _ _ (h e (List.mem_cons_self ..))

theorem writes_byte_frame (api : API α Store Byte) (events : List Write)
    (m : Store) (b : Nat) (h : ∀ e ∈ events, b < e.address ∨ e.address+8 ≤ b) :
    api.byte (runWrites api events m) b = api.byte m b := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change api.byte (runWrites api es (api.write m e.address e.value)) b = _
    rw [ih _ (fun q hq => h q (List.mem_cons_of_mem _ hq))]
    exact api.byte_frame m _ _ _ (h e (List.mem_cons_self ..))

theorem writes_readback (api : API α Store Byte) (events : List Write)
    (m : Store) (a : Nat) (v : BitVec 64)
    (present : ∃ e ∈ events, e.address=a ∧ e.value=v)
    (only : ∀ e ∈ events, (e.address=a ∧ e.value=v) ∨ e.address+8 ≤ a ∨ a+8 ≤ e.address) :
    api.read (runWrites api events m) a = v := by
  induction events generalizing m with
  | nil => simp at present
  | cons e es ih =>
    change api.read (runWrites api es (api.write m e.address e.value)) a = v
    by_cases tail : ∃ q ∈ es, q.address=a ∧ q.value=v
    · exact ih _ tail (fun q hq => only q (List.mem_cons_of_mem _ hq))
    · have head : e.address=a ∧ e.value=v := by
        obtain ⟨q,hq,ha,hv⟩ := present
        rcases List.mem_cons.mp hq with he | ht
        · simpa [he] using And.intro ha hv
        · exact (tail ⟨q,ht,ha,hv⟩).elim
      rw [writes_read_frame api es _ a]
      · rw [head.1,head.2,api.read_same]
      · intro q hq
        rcases only q (List.mem_cons_of_mem _ hq) with same | apart
        · exact (tail ⟨q,hq,same⟩).elim
        · exact apart

abbrev Location (p : Parameters) := Nat × Fin p.block

def schedule (p : Parameters) : List (Location p) :=
  (List.range (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid p)).flatMap fun pid => (List.finRange p.block).map fun lane => (pid,lane)

def global (p : Parameters) (x : Location p) : Nat := x.1*p.block+x.2.val

def rowValue {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α) (x : Location p) : Int :=
  (rounds P .fp32 p (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p s s x.1) (count p)).output (x.2,PUnit.unit)

/-- Each active epilogue writes the source recurrence's actual integer value.
The source-execution theorem below separately proves this exact recurrence. -/
def events {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α)
    (output : Tensor) (xs : List (Location p)) : List Write :=
  xs.filterMap fun x => if global p x < p.batches then
    some ⟨output.base+8*global p x,BitVec.ofInt 64 (rowValue P p s x)⟩ else none

def run {α : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α) (p : Parameters)
    (s : State α) (output : Tensor) (m : Store) : Store :=
  runWrites api (events P p s output (schedule p)) m

theorem source_execution {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (s : State α) (x : Location p)
    (seeds : SeedDomain p (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p s s x.1)) :
    ∃ final, FP.Structural.exec (backend rest P) (source .fp32 p) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p s s x.1) = some final ∧
      (global p x < p.batches → final.mem p.output (global p x) = Cell.mk .int (rowValue P p s x)) := by
  obtain ⟨final,he,_,hv,_⟩ := correctness rest P .fp32 p (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState p s s x.1) seeds
  refine ⟨final,he,?_⟩
  intro hx
  exact hv (x.2,PUnit.unit) hx

theorem row_value {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α) (x : Location p) :
    rowValue P p s x = VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value P .fp32 p s (global p x) := by
  have hd : global p x / p.block = x.1 := by
    unfold global
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ p.blockDomain.1,
      Nat.div_eq_of_lt x.2.isLt, Nat.zero_add]
  have hl : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane p (global p x) = (x.2,PUnit.unit) := by
    apply Prod.ext
    · apply Fin.ext
      simp [VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.lane,global,Nat.add_mod,Nat.mod_eq_of_lt x.2.isLt]
    · exact Subsingleton.elim _ _
  simp only [VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value,hd,hl,rowValue]

theorem schedule_cover (p : Parameters) (o : Nat) (ho : o < p.batches) :
    ∃ x ∈ schedule p, global p x = o := by
  let lane : Fin p.block := ⟨o%p.block,Nat.mod_lt _ p.blockDomain.1⟩
  have hg : o/p.block < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid p := by
    apply (Nat.div_lt_iff_lt_mul p.blockDomain.1).mpr
    have := VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid_covers p
    omega
  refine ⟨(o/p.block,lane),?_,?_⟩
  · simp [schedule,hg,lane]
  · unfold global
    dsimp [lane]
    rw [Nat.mul_comm]
    exact Nat.div_add_mod o p.block

theorem mem_events {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α)
    (output : Tensor) (xs : List (Location p)) (e : Write) :
    e ∈ events P p s output xs ↔ ∃ x ∈ xs, global p x < p.batches ∧
      e = ⟨output.base+8*global p x,BitVec.ofInt 64 (rowValue P p s x)⟩ := by
  simp only [events,List.mem_filterMap]
  constructor
  · rintro ⟨x,hx,he⟩
    split at he
    · exact ⟨x,hx,‹_›,(Option.some.inj he).symm⟩
    · simp at he
  · rintro ⟨x,hx,ha,rfl⟩
    exact ⟨x,hx,by simp [ha]⟩

theorem readback {α : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (s : State α) (output : Tensor) (m : Store) (o : Nat) (ho : o < p.batches) :
    api.read (run api P p s output m) (output.base+8*o) =
      BitVec.ofInt 64 (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value P .fp32 p s o) := by
  apply writes_readback
  · obtain ⟨x,hx,he⟩ := schedule_cover p o ho
    refine ⟨⟨output.base+8*o,BitVec.ofInt 64 (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value P .fp32 p s o)⟩,?_,rfl,rfl⟩
    apply (mem_events P p s output (schedule p) _).mpr
    refine ⟨x,hx,by simpa [he] using ho,?_⟩
    rw [row_value,he]
  · intro e he
    obtain ⟨x,hx,ha,rfl⟩ := (mem_events P p s output (schedule p) e).mp he
    by_cases h : global p x = o
    · left
      rw [row_value,h]
      exact ⟨rfl,rfl⟩
    · right
      dsimp only
      omega

theorem byte_frame {α : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (s : State α) (output : Tensor) (m : Store) (b : Nat)
    (outside : b < output.base ∨ output.base+8*p.batches ≤ b) :
    api.byte (run api P p s output m) b = api.byte m b := by
  apply writes_byte_frame
  intro e he
  obtain ⟨x,hx,ha,rfl⟩ := (mem_events P p s output (schedule p) e).mp he
  dsimp only
  omega

theorem active_store_bounds {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters) (s : State α)
    (output : Tensor) (fits : output.base+8*p.batches ≤ output.allocationEnd)
    (endFits : output.allocationEnd ≤ 2^64) (e : Write) (he : e ∈ events P p s output (schedule p)) :
    output.base ≤ e.address ∧ e.address+8 ≤ output.allocationEnd ∧ e.address+8 ≤ 2^64 := by
  obtain ⟨x,hx,ha,rfl⟩ := (mem_events P p s output (schedule p) e).mp he
  dsimp only
  omega

#axiomsClean source_execution
#axiomsClean readback
#axiomsClean byte_frame
#axiomsClean active_store_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Physical
