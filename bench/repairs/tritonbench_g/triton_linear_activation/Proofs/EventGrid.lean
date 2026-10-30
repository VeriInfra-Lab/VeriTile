import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Stores
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

def owns (s : Shape) (c : Fin 98) (pid offset : Nat) : Prop :=
  offset<s.m*s.n ∧
  tileRow (s.rows c) (s.cols c) 8 pid=(offset/max s.n 1)/(configs c).bm ∧
  tileCol (s.rows c) (s.cols c) 8 pid=(offset%max s.n 1)/(configs c).bn
instance (s : Shape) (c : Fin 98) (pid offset : Nat) : Decidable (owns s c pid offset) := by
  unfold owns; infer_instance

def atomicWrites (s : Shape) (b : Buffers) (c : Fin 98) (pid offset : Nat) (p : Address) : Prop :=
  writes s b c pid p ∧ p.2=offset
instance (s : Shape) (b : Buffers) (c : Fin 98) (pid offset : Nat) : DecidablePred (atomicWrites s b c pid offset) := by
  intro p; unfold atomicWrites; infer_instance

def atomicCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String)
    (pid offset : Nat) : Code α Unit :=
  if owns s c pid offset then laneCode s b c mode name offset else pure ()
def atomicStep (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode)
    (name : Option String) (pid offset : Nat) (memory : Memory α) : Memory α :=
  if owns s c pid offset then laneStep a s b c mode name offset memory else memory

 theorem atomicStep_form (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode)
    (name : Option String) (pid offset : Nat) (memory : Memory α) (sep : Separate s b) :
    atomicStep a s b c mode name pid offset memory=
      Schedule.put (atomicWrites s b c pid)
        (fun _ p => programValue a s b c mode name memory pid p) offset memory := by
  classical
  unfold atomicStep
  split_ifs with ho
  · rw [laneStep_form a s b c mode name offset memory sep]
    funext p
    simp only [Schedule.put]
    have he : atomicWrites s b c pid offset p ↔ b.writesStorage p ∧ p.2=offset := by
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

def atomicRun (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode)
    (name : Option String) (pid : Nat) (memory : Memory α) : Memory α :=
  Schedule.run (atomicStep a s b c mode name pid) (List.range (s.m*s.n)) memory

 theorem atomicRun_frozen (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode)
    (name : Option String) (pid : Nat) (entry : Memory α) (sep : Separate s b) :
    atomicRun a s b c mode name pid entry=
      Schedule.run (Schedule.put (atomicWrites s b c pid)
        (fun _ p => programValue a s b c mode name entry pid p)) (List.range (s.m*s.n)) entry := by
  apply Schedule.freeze_reads (reads:=fun _ => b.reads s) (writes:=atomicWrites s b c pid) (entry:=entry)
  · intro off current same
    rw [atomicStep_form a s b c mode name pid off current sep,
      programValue_reads a s b c mode name entry current same]
  · intro off current p outside
    rw [atomicStep_form a s b c mode name pid off current sep]
    exact if_neg outside
  · exact List.nodup_range
  · intro p hp q hq hne loc read write
    exact sep.1 loc read write.1.1
  · intro p hp loc read; rfl

/-- Whole vector store ownership is derived from the two scalar stores per
active lane. Neither a final-memory assumption nor a dot-result callback occurs. -/
theorem atomicRun_eq_program (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (pid : Nat) (memory : Memory α) (sep : Separate s b) :
    atomicRun a s b c mode name pid memory=programStep a s b c mode name pid memory := by
  rw [atomicRun_frozen a s b c mode name pid memory sep]
  funext p
  by_cases hp : writes s b c pid p
  · rw [Schedule.fixed_output (atomicWrites s b c pid)
      (fun _ p => programValue a s b c mode name memory pid p)
      (List.range (s.m*s.n)) memory p.2 p ⟨hp,rfl⟩]
    · simp only [List.mem_range,hp.2.1,if_pos,programStep,Schedule.put,hp]
    · intro q hq h; exact h.2.symm
  · rw [Schedule.fixed_frame (atomicWrites s b c pid)
      (fun _ p => programValue a s b c mode name memory pid p)
      (List.range (s.m*s.n)) memory p (by intro q hq h; exact hp h.1)]
    simp [programStep,Schedule.put,hp]

def sequenceCode (f : Nat → Code α Unit) : List Nat → Code α Unit
  | [] => pure ()
  | q::tail => do
      let _ ← f q
      sequenceCode f tail

 theorem eval_sequenceCode (a : Arithmetic α) (f : Nat → Code α Unit)
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
def programCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String)
    (pid : Nat) : Code α Unit :=
  sequenceCode (atomicCode s b c mode name pid) (List.range (s.m*s.n))
def gridCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (name : Option String) : Code α Unit :=
  sequenceCode (programCode s b c mode name) (List.range (s.programs c))

@[simp] theorem eval_atomicCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (pid offset : Nat) (memory : Memory α) :
    (atomicCode s b c mode name pid offset).eval a memory=
      ((),atomicStep a s b c mode name pid offset memory) := by
  unfold atomicCode atomicStep
  split_ifs <;> simp [pure]

 theorem eval_programCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (pid : Nat) (memory : Memory α) (sep : Separate s b) :
    (programCode s b c mode name pid).eval a memory=((),programStep a s b c mode name pid memory) := by
  unfold programCode
  rw [eval_sequenceCode a _ _ (fun q m => eval_atomicCode a s b c mode name pid q m)]
  exact congrArg (fun m => ((),m)) (atomicRun_eq_program a s b c mode name pid memory sep)

 theorem eval_gridCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (memory : Memory α) (sep : Separate s b) :
    (gridCode s b c mode name).eval a memory=((),run a s b c mode name memory) := by
  exact eval_sequenceCode a _ _ (fun q m => eval_programCode a s b c mode name q m sep) _ memory

 theorem native_grid (abi : LocalEventABI α σ a) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (state : σ) (sep : Separate s b) :
    abi.observe ((gridCode s b c mode name).native abi state).2=
      run a s b c mode name (abi.observe state) := by
  have h := Code.native_refines abi (gridCode s b c mode name) state
  rw [eval_gridCode a s b c mode name _ sep] at h
  exact congrArg Prod.snd h

#axiomsClean atomicStep_form
#axiomsClean atomicRun_eq_program
#axiomsClean eval_gridCode
#axiomsClean native_grid
end
end VeriTile.Bench.Repairs.TritonLinearActivation
