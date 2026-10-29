import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Events

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
abbrev Schedule := @CrossEntropy1.Schedule.run

/-- Actual source masks, wrapped row/column and safe-K pointer coordinates. -/
def loadACode (p : Problem) (b : Buffers) (tile iteration : Nat) (row : Fin 128) (k : Fin 64) : Code α α :=
  if p.k iteration k.val < p.K then
    emit (.read (b.A.address ((p.row tile row.val)%p.M) (clamp p.K (p.k iteration k.val))))
  else emit .zero

def loadBCode (p : Problem) (b : Buffers) (tile iteration : Nat) (k : Fin 64) (column : Fin 256) : Code α α :=
  if p.k iteration k.val < p.K then
    emit (.read (b.B.address (clamp p.K (p.k iteration k.val)) ((p.column tile column.val)%p.N)))
  else emit .zero

@[simp] theorem eval_loadA (ops : Operations α) (p : Problem) (b : Buffers) (tile iteration : Nat)
    (row : Fin 128) (k : Fin 64) (memory : Memory α) :
    (loadACode p b tile iteration row k).eval ops memory =
      (Source.loadA (sourceArithmetic ops) p b memory tile iteration row k,memory) := by
  unfold loadACode Source.loadA
  split_ifs <;> rfl

@[simp] theorem eval_loadB (ops : Operations α) (p : Problem) (b : Buffers) (tile iteration : Nat)
    (k : Fin 64) (column : Fin 256) (memory : Memory α) :
    (loadBCode p b tile iteration k column).eval ops memory =
      (Source.loadB (sourceArithmetic ops) p b memory tile iteration k column,memory) := by
  unfold loadBCode Source.loadB
  split_ifs <;> rfl

def blockCode (p : Problem) (b : Buffers) (tile iteration : Nat)
    (row : Fin 128) (column : Fin 256) (seed : α) : Code α α :=
  foldCode (fun group z => do
    let xs ← gather (fun lane : Fin 8 => loadACode p b tile iteration row ⟨group.val*8+lane.val,by omega⟩)
    let ys ← gather (fun lane : Fin 8 => loadBCode p b tile iteration ⟨group.val*8+lane.val,by omega⟩ column)
    emit (.instruction xs ys z)) (List.finRange 8) seed

theorem eval_block (ops : Operations α) (p : Problem) (b : Buffers) (tile iteration : Nat)
    (row : Fin 128) (column : Fin 256) (seed : α) (memory : Memory α) :
    (blockCode p b tile iteration row column seed).eval ops memory =
      (block ops (Source.loadA (sourceArithmetic ops) p b memory tile iteration row)
        (fun k => Source.loadB (sourceArithmetic ops) p b memory tile iteration k column) seed,memory) := by
  unfold blockCode block
  apply eval_fold
  intro group z
  simp only [bind,eval_bind,eval_gather ops _ _ memory (fun i => eval_loadA ops p b tile iteration row _ memory),Prod.fst,Prod.snd]
  simp only [eval_gather ops _ _ memory (fun i => eval_loadB ops p b tile iteration _ column memory),Prod.fst,Prod.snd,eval_emit,Event.eval]

def accumulatorCode (p : Problem) (b : Buffers) (tile : Nat) (row : Fin 128) (column : Fin 256) : Nat → Code α α
  | 0 => emit .zero
  | count+1 => do
    let acc ← accumulatorCode p b tile row column count
    blockCode p b tile count row column acc

theorem eval_accumulator (ops : Operations α) (p : Problem) (b : Buffers) (tile count : Nat)
    (row : Fin 128) (column : Fin 256) (memory : Memory α) :
    (accumulatorCode p b tile row column count).eval ops memory =
      (Source.accumulator (sourceArithmetic ops) p b memory tile count row column,memory) := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [accumulatorCode,bind,eval_bind,ih,Prod.fst,Prod.snd,eval_block,Source.accumulator,sourceArithmetic,Storage.blockK]
    rfl

def activationCode (enabled : Bool) (value : α) : Code α α :=
  if enabled then do
    let nonnegative ← emit (.nonnegative value)
    let negative ← emit (.slopeMultiply value)
    pure (if nonnegative then value else negative)
  else pure value

theorem eval_activation (ops : Operations α) (enabled : Bool) (value : α) (memory : Memory α) :
    (activationCode enabled value).eval ops memory = (activate ops enabled value,memory) := by
  cases enabled <;> simp [activationCode,activate,bind,eval_bind,Event.eval,pure]

def laneValue (p : Problem) (b : Buffers) (tile : Nat) (lane : Source.Lane) : Code α α := do
  let acc ← accumulatorCode p b tile lane.1 lane.2 p.kTiles
  let active ← activationCode p.enabled acc
  emit (.castHalf active)

theorem eval_lane (ops : Operations α) (p : Problem) (b : Buffers) (tile : Nat)
    (lane : Source.Lane) (memory : Memory α) :
    (laneValue p b tile lane).eval ops memory =
      ((sourceArithmetic ops).cast .fp16 ((sourceArithmetic ops).activation p.enabled
        (Source.accumulator (sourceArithmetic ops) p b memory tile p.kTiles lane.1 lane.2)),memory) := by
  simp only [laneValue,bind,eval_bind,eval_accumulator,Prod.fst,Prod.snd,eval_activation,eval_emit,Event.eval,sourceArithmetic]

def valueCode (p : Problem) (b : Buffers) (tile : Nat) (point : Address) : Code α α :=
  if h : Source.writes p b tile point then laneValue p b tile (Classical.choose h) else emit .zero

theorem eval_value (ops : Operations α) (p : Problem) (b : Buffers) (tile : Nat)
    (point : Address) (memory : Memory α) :
    (valueCode p b tile point).eval ops memory = (Source.value (sourceArithmetic ops) p b memory tile point,memory) := by
  unfold valueCode Source.value
  split_ifs <;> simp only [eval_lane,eval_emit,Event.eval,sourceArithmetic]

def atomicWrites (p : Problem) (b : Buffers) (tile index : Nat) (point : Address) : Prop :=
  Source.writes p b tile point ∧ point.2 = b.C.offset+index
instance (p : Problem) (b : Buffers) (tile index : Nat) : DecidablePred (atomicWrites p b tile index) := Classical.decPred _

def atomicCode (p : Problem) (b : Buffers) (tile index : Nat) : Code α Unit :=
  let point := (b.C.storage,b.C.offset+index)
  if Source.writes p b tile point then do
    let value ← valueCode p b tile point
    emit (.storeHalf point value)
  else pure ()

def atomicStep (ops : Operations α) (p : Problem) (b : Buffers) (tile index : Nat) (memory : Memory α) : Memory α :=
  CrossEntropy1.Schedule.put (atomicWrites p b tile) (fun _ point => Source.value (sourceArithmetic ops) p b memory tile point) index memory

theorem writes_storage (p : Problem) (b : Buffers) (tile : Nat) (point : Address)
    (h : Source.writes p b tile point) : point.1 = b.C.storage := by
  obtain ⟨lane,_,address⟩ := h
  exact (congrArg Prod.fst address).symm

theorem eval_atomic (ops : Operations α) (p : Problem) (b : Buffers) (tile index : Nat) (memory : Memory α) :
    (atomicCode p b tile index).eval ops memory = ((),atomicStep ops p b tile index memory) := by
  unfold atomicCode
  dsimp only
  split_ifs with active
  · simp only [bind,eval_bind,eval_value,Prod.fst,Prod.snd,eval_emit,Event.eval]
    congr 1
    funext point
    have eq : atomicWrites p b tile index point ↔ point = (b.C.storage,b.C.offset+index) := by
      constructor
      · intro h; exact Prod.ext (writes_storage p b tile point h.1) h.2
      · intro h; subst point; exact ⟨active,rfl⟩
    simp only [write,atomicStep,CrossEntropy1.Schedule.put,eq]
    split_ifs with same
    · subst point; rfl
    · rfl
  · simp only [pure,eval_pure]
    congr 1
    funext point
    have no : ¬atomicWrites p b tile index point := by
      intro h
      have same : point = (b.C.storage,b.C.offset+index) := Prod.ext (writes_storage p b tile point h.1) h.2
      exact active (same ▸ h.1)
    simp only [atomicStep,CrossEntropy1.Schedule.put,if_neg no]

def sequenceCode (f : Nat → Code α Unit) : List Nat → Code α Unit
  | [] => pure ()
  | i::rest => do let _ ← f i; sequenceCode f rest

theorem eval_sequence (ops : Operations α) (f : Nat → Code α Unit)
    (step : Nat → Memory α → Memory α) (law : ∀ i m, (f i).eval ops m = ((),step i m))
    (tasks : List Nat) (memory : Memory α) :
    (sequenceCode f tasks).eval ops memory = ((),Schedule step tasks memory) := by
  induction tasks generalizing memory with
  | nil => rfl
  | cons i rest ih => simp only [sequenceCode,bind,eval_bind,law,Prod.fst,Prod.snd,ih]; rfl

/-- Independent active stores may be reindexed by the actual contiguous
output offset; input storage is preserved after every individual store. -/
theorem atomic_frozen (ops : Operations α) (p : Problem) (b : Buffers) (layout : Layout p b)
    (tile : Nat) (memory : Memory α) :
    Schedule (atomicStep ops p b tile) (List.range (p.M*p.N)) memory =
      Schedule (CrossEntropy1.Schedule.put (atomicWrites p b tile)
        (fun _ point => Source.value (sourceArithmetic ops) p b memory tile point)) (List.range (p.M*p.N)) memory := by
  apply CrossEntropy1.Schedule.freeze_reads (reads:=fun _ => Source.reads b) (writes:=atomicWrites p b tile) (entry:=memory)
  · intro index current same
    simp only [atomicStep,Source.value_stable (sourceArithmetic ops) p b memory current same]
  · intro index current point outside
    exact if_neg outside
  · exact List.nodup_range
  · intro i hi j hj different point read written
    exact Source.read_separate p b layout point read tile written.1
  · intros; rfl

theorem written_offset (p : Problem) (b : Buffers) (layout : Layout p b) (tile : Nat) (point : Address)
    (written : Source.writes p b tile point) :
    b.C.offset ≤ point.2 ∧ point.2-b.C.offset < p.M*p.N := by
  obtain ⟨lane,active,address⟩ := written
  rw [Source.output_address p b tile lane active] at address
  have eq := congrArg Prod.snd address
  have positive : 1 ≤ p.N := by have := active.2; omega
  simp only [Tensor.address,layout.outputStrides.1,layout.outputStrides.2,Nat.mul_one,max_eq_left positive] at eq
  have bound := Nat.mul_le_mul_right p.N (Nat.succ_le_of_lt active.1)
  simp only [Nat.succ_mul] at bound
  have column := active.2
  constructor <;> omega

theorem atomic_complete (ops : Operations α) (p : Problem) (b : Buffers) (layout : Layout p b)
    (tile : Nat) (memory : Memory α) :
    Schedule (atomicStep ops p b tile) (List.range (p.M*p.N)) memory = Source.step (sourceArithmetic ops) p b tile memory := by
  rw [atomic_frozen ops p b layout tile memory]
  funext point
  unfold Schedule
  by_cases written : Source.writes p b tile point
  · have bound := written_offset p b layout tile point written
    rw [CrossEntropy1.Schedule.fixed_output (atomicWrites p b tile)
      (fun _ point => Source.value (sourceArithmetic ops) p b memory tile point)
      (List.range (p.M*p.N)) memory (point.2-b.C.offset) point ⟨written,by omega⟩]
    · simp only [List.mem_range,bound.2,if_true,Source.step,CrossEntropy1.Schedule.put,written]
    · intro index hi write; have eq := write.2; omega
  · rw [CrossEntropy1.Schedule.fixed_frame (atomicWrites p b tile)
      (fun _ point => Source.value (sourceArithmetic ops) p b memory tile point)
      (List.range (p.M*p.N)) memory point (by intro index hi write; exact written write.1)]
    simp only [Source.step,CrossEntropy1.Schedule.put,if_neg written]

def programCode (p : Problem) (b : Buffers) (tile : Nat) : Code α Unit :=
  sequenceCode (atomicCode p b tile) (List.range (p.M*p.N))
def gridCode (p : Problem) (b : Buffers) : Code α Unit :=
  sequenceCode (programCode p b) (Source.tiles p)

theorem eval_program (ops : Operations α) (p : Problem) (b : Buffers) (layout : Layout p b)
    (tile : Nat) (memory : Memory α) :
    (programCode p b tile).eval ops memory = ((),Source.step (sourceArithmetic ops) p b tile memory) := by
  rw [programCode,eval_sequence ops _ _ (eval_atomic ops p b tile),atomic_complete ops p b layout]

theorem eval_grid (ops : Operations α) (p : Problem) (b : Buffers) (layout : Layout p b) (memory : Memory α) :
    (gridCode p b).eval ops memory = ((),Source.run (sourceArithmetic ops) p b memory) :=
  eval_sequence ops _ _ (eval_program ops p b layout) _ memory

theorem native_grid (abi : LocalABI α σ ops) (p : Problem) (b : Buffers) (layout : Layout p b) (state : σ) :
    abi.observe ((gridCode p b).native abi state).2 = Source.run (sourceArithmetic ops) p b (abi.observe state) := by
  have h := native_refines abi (gridCode p b) state
  rw [eval_grid ops p b layout] at h
  exact congrArg Prod.snd h

#axiomsClean eval_grid
#axiomsClean native_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
