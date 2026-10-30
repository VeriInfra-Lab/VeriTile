import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Events
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

def laneCode (s : Shape) (b : Buffers) (c : Fin 15) (offset : Nat) : Code α β Unit := do
  let z ← accumulationCode s b c (offset/max s.n 1) (offset%max s.n 1)
  emit (.store (b.outRegion,offset) z)

def laneStep (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (offset : Nat) (memory : Memory α) : Memory α :=
  writeWord memory (b.outRegion,offset)
    (a.encode (sourceAccumulation a s b c memory (offset/max s.n 1) (offset%max s.n 1)))

@[simp] theorem eval_laneCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (offset : Nat) (memory : Memory α) :
    (laneCode s b c offset).eval a memory=((),laneStep a s b c offset memory) := by
  simp only [laneCode,bind,eval_bind,eval_accumulationCode,Prod.fst,Prod.snd,eval_emit,Event.eval,laneStep]

theorem laneStep_form (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (offset : Nat) (memory : Memory α) (legal : s.legal c) (sep : Separate b) :
    laneStep a s b c offset memory = fun p =>
      if p.1=b.outRegion ∧ p.2=offset then programValue a s b c memory 0 p else memory p := by
  funext p
  unfold laneStep writeWord programValue
  by_cases he : p=(b.outRegion,offset)
  · subst p; simp
  · have hn : ¬(p.1=b.outRegion ∧ p.2=offset) := by
      intro h; exact he (Prod.ext h.1 h.2)
    simp only [he,hn,ite_false]

def owns (s : Shape) (c : Fin 15) (pid offset : Nat) : Prop :=
  offset<s.m*s.n ∧
  tileRow (s.rows c) (s.cols c) (configs c).gm pid=(offset/max s.n 1)/(configs c).bm ∧
  tileCol (s.rows c) (s.cols c) (configs c).gm pid=(offset%max s.n 1)/(configs c).bn
instance (s : Shape) (c : Fin 15) (pid offset : Nat) : Decidable (owns s c pid offset) := by
  unfold owns; infer_instance

def atomicWrites (s : Shape) (b : Buffers) (c : Fin 15) (pid offset : Nat) (p : Address) : Prop :=
  writes s b c pid p ∧ p.2=offset
instance (s : Shape) (b : Buffers) (c : Fin 15) (pid offset : Nat) : DecidablePred (atomicWrites s b c pid offset) := by
  intro p; unfold atomicWrites; infer_instance

def atomicCode (s : Shape) (b : Buffers) (c : Fin 15) 
    (pid offset : Nat) : Code α β Unit :=
  if owns s c pid offset then laneCode s b c offset else pure ()
def atomicStep (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)  (pid offset : Nat) (memory : Memory α) : Memory α :=
  if owns s c pid offset then laneStep a s b c offset memory else memory

 theorem atomicStep_form (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)  (pid offset : Nat) (memory : Memory α) (legal : s.legal c) (sep : Separate b) :
    atomicStep a s b c pid offset memory=
      Schedule.put (atomicWrites s b c pid)
        (fun _ p => programValue a s b c memory pid p) offset memory := by
  classical
  unfold atomicStep
  split_ifs with ho
  · rw [laneStep_form a s b c offset memory legal sep]
    funext p
    simp only [Schedule.put]
    have he : atomicWrites s b c pid offset p ↔ p.1=b.outRegion ∧ p.2=offset := by
      unfold atomicWrites writes
      constructor
      · intro h; exact ⟨h.1.1,h.2⟩
      · intro h; refine ⟨⟨h.1,?_⟩,h.2⟩; rw [h.2]; exact ho
    simp only [he]
    rfl
  · funext p
    have hn : ¬atomicWrites s b c pid offset p := by
      intro h
      apply ho
      have k := h.1.2
      simpa only [h.2] using k
    simp [Schedule.put,hn]

def atomicRun (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)  (pid : Nat) (memory : Memory α) : Memory α :=
  Schedule.run (atomicStep a s b c pid) (List.range (s.m*s.n)) memory

 theorem atomicRun_frozen (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)  (pid : Nat) (entry : Memory α) (legal : s.legal c) (sep : Separate b) :
    atomicRun a s b c pid entry=
      Schedule.run (Schedule.put (atomicWrites s b c pid)
        (fun _ p => programValue a s b c entry pid p)) (List.range (s.m*s.n)) entry := by
  apply Schedule.freeze_reads (reads:=fun _ => b.reads) (writes:=atomicWrites s b c pid) (entry:=entry)
  · intro off current same
    rw [atomicStep_form a s b c pid off current legal sep,
      programValue_reads a s b c legal entry current same]
  · intro off current p outside
    rw [atomicStep_form a s b c pid off current legal sep]
    exact if_neg outside
  · exact List.nodup_range
  · intro p hp q hq hne loc read write
    exact separate_read_write b sep loc read write.1.1
  · intro p hp loc read; rfl

/-- Whole vector store ownership is derived from the scalar store per
active lane. Neither a final-memory assumption nor a dot-result callback occurs. -/
theorem atomicRun_eq_program (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
     (pid : Nat) (memory : Memory α) (legal : s.legal c) (sep : Separate b) :
    atomicRun a s b c pid memory=programStep a s b c pid memory := by
  rw [atomicRun_frozen a s b c pid memory legal sep]
  funext p
  by_cases hp : writes s b c pid p
  · rw [Schedule.fixed_output (atomicWrites s b c pid)
      (fun _ p => programValue a s b c memory pid p)
      (List.range (s.m*s.n)) memory p.2 p ⟨hp,rfl⟩]
    · simp only [List.mem_range,hp.2.1,if_pos,programStep,Schedule.put,hp]
    · intro q hq h; exact h.2.symm
  · rw [Schedule.fixed_frame (atomicWrites s b c pid)
      (fun _ p => programValue a s b c memory pid p)
      (List.range (s.m*s.n)) memory p (by intro q hq h; exact hp h.1)]
    simp [programStep,Schedule.put,hp]

def sequenceCode (f : Nat → Code α β Unit) : List Nat → Code α β Unit
  | [] => pure ()
  | q::tail => do
      let _ ← f q
      sequenceCode f tail

 theorem eval_sequenceCode (a : Arithmetic α β) (f : Nat → Code α β Unit)
    (step : Nat → Memory α → Memory α)
    (localRule : ∀ q m,(f q).eval a m=((),step q m)) (tasks : List Nat) (m : Memory α) :
    (sequenceCode f tasks).eval a m=((),Schedule.run step tasks m) := by
  induction tasks generalizing m with
  | nil => rfl
  | cons q tail ih =>
    simp only [sequenceCode,bind,eval_bind,localRule,Prod.fst,Prod.snd,ih]
    rfl

/-- The outer order is the exact grouped pid order. The active lane domain is
reindexed by canonical offset; source_store_ownership/unique and the inverse
coordinate laws establish this harmless ordering of independent lane stores. -/
def programCode (s : Shape) (b : Buffers) (c : Fin 15) 
    (pid : Nat) : Code α β Unit :=
  sequenceCode (atomicCode s b c pid) (List.range (s.m*s.n))
def gridCode (s : Shape) (b : Buffers) (c : Fin 15)  : Code α β Unit :=
  sequenceCode (programCode s b c) (List.range (s.programs c))

@[simp] theorem eval_atomicCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
     (pid offset : Nat) (memory : Memory α) :
    (atomicCode s b c pid offset).eval a memory=
      ((),atomicStep a s b c pid offset memory) := by
  unfold atomicCode atomicStep
  split_ifs <;> simp [pure]

 theorem eval_programCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
     (pid : Nat) (memory : Memory α) (legal : s.legal c) (sep : Separate b) :
    (programCode s b c pid).eval a memory=((),programStep a s b c pid memory) := by
  unfold programCode
  rw [eval_sequenceCode a _ _ (fun q m => eval_atomicCode a s b c pid q m)]
  exact congrArg (fun m => ((),m)) (atomicRun_eq_program a s b c pid memory legal sep)

 theorem eval_gridCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
     (memory : Memory α) (legal : s.legal c) (sep : Separate b) :
    (gridCode s b c).eval a memory=((),run a s b c memory) := by
  exact eval_sequenceCode a _ _ (fun q m => eval_programCode a s b c q m legal sep) _ memory

 theorem native_grid (abi : LocalEventABI α β σ a) (s : Shape) (b : Buffers) (c : Fin 15)
     (state : σ) (legal : s.legal c) (sep : Separate b) :
    abi.observe ((gridCode s b c).native abi state).2=
      run a s b c (abi.observe state) := by
  have h := Code.native_refines abi (gridCode s b c) state
  rw [eval_gridCode a s b c _ legal sep] at h
  exact congrArg Prod.snd h

#axiomsClean atomicStep_form
#axiomsClean atomicRun_eq_program
#axiomsClean eval_gridCode
#axiomsClean native_grid
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
