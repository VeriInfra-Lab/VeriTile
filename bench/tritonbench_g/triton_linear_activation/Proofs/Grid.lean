import bench.tritonbench_g.triton_linear_activation.Proofs.Source
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def writes (s : Shape) (b : Buffers) (c : Fin 98) (pid : Nat) (p : Address) : Prop :=
  b.writesStorage p ∧ p.2<s.m*s.n ∧
  tileRow (s.rows c) (s.cols c) 8 pid=(p.2/max s.n 1)/(configs c).bm ∧
  tileCol (s.rows c) (s.cols c) 8 pid=(p.2%max s.n 1)/(configs c).bn
instance (s : Shape) (b : Buffers) (c : Fin 98) (pid : Nat) : DecidablePred (writes s b c pid) := by
  intro p
  unfold writes Buffers.writesStorage
  infer_instance

def programValue (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (memory : Memory α) (_pid : Nat) (p : Address) : α :=
  let z := sourceAccumulation a s b c mode memory (p.2/max s.n 1) (p.2%max s.n 1)
  a.encode mode (if p.1=b.out then activation a name z else z)

def programStep (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (pid : Nat) (memory : Memory α) : Memory α :=
  Schedule.put (writes s b c) (programValue a s b c mode name memory) pid memory

def run (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (memory : Memory α) : Memory α :=
  Schedule.run (programStep a s b c mode name) (List.range (s.programs c)) memory

 theorem output_coordinates (s : Shape) (m n : Nat) (hn : n<s.n) :
    outputOffset s m n/max s.n 1=m ∧ outputOffset s m n%max s.n 1=n := by
  have hp : 0<max s.n 1 := by omega
  have hc : n<max s.n 1 := lt_of_lt_of_le hn (le_max_left _ _)
  unfold outputOffset
  constructor
  · rw [Nat.mul_comm m,Nat.mul_add_div hp,Nat.div_eq_of_lt hc,Nat.add_zero]
  · rw [Nat.mul_comm m,Nat.mul_add_mod,Nat.mod_eq_of_lt hc]

 theorem output_injective (s : Shape) (m n m' n' : Nat) (hn : n<s.n) (hn' : n'<s.n)
    (he : outputOffset s m n=outputOffset s m' n') : m=m' ∧ n=n' := by
  have h := output_coordinates s m n hn
  have h' := output_coordinates s m' n' hn'
  constructor
  · rw [←h.1,←h'.1,he]
  · rw [←h.2,←h'.2,he]

 theorem source_store_ownership (s : Shape) (b : Buffers) (c : Fin 98)
    (pid i j : Nat) (hi : i<(configs c).bm) (hj : j<(configs c).bn)
    (hm : row s c pid i<s.m) (hn : col s c pid j<s.n) (p : Nat)
    (hw : p=b.out ∨ b.saved=some p) :
    writes s b c pid (p,outputOffset s (row s c pid i) (col s c pid j)) := by
  have pos : 0<s.n := by omega
  have maxn : max s.n 1=s.n := max_eq_left pos
  have coords := output_coordinates s (row s c pid i) (col s c pid j) hn
  refine ⟨hw,?_,?_,?_⟩
  · simp only [outputOffset,Prod.snd,maxn]
    have h := Nat.mul_le_mul_right s.n (Nat.succ_le_of_lt hm)
    nlinarith
  · simp only [Prod.snd]
    rw [coords.1]
    unfold row
    have bm : 0<(configs c).bm := by have := config_bounds c; omega
    rw [Nat.mul_comm _ (configs c).bm,Nat.mul_add_div bm,Nat.div_eq_of_lt hi,Nat.add_zero]
  · simp only [Prod.snd]
    rw [coords.2]
    unfold col
    have bn : 0<(configs c).bn := by have := config_bounds c; omega
    rw [Nat.mul_comm _ (configs c).bn,Nat.mul_add_div bn,Nat.div_eq_of_lt hj,Nat.add_zero]

 theorem source_store_unique (s : Shape) (c : Fin 98) (pid i j i' j' : Nat)
    (hn : col s c pid j<s.n) (hn' : col s c pid j'<s.n)
    (he : outputOffset s (row s c pid i) (col s c pid j)=
      outputOffset s (row s c pid i') (col s c pid j')) : i=i' ∧ j=j' := by
  have h := output_injective s _ _ _ _ hn hn' he
  exact ⟨Nat.add_left_cancel h.1,Nat.add_left_cancel h.2⟩

 theorem writes_unique (s : Shape) (b : Buffers) (c : Fin 98) (p q : Nat)
    (hp : p<s.programs c) (hq : q<s.programs c) (loc : Address)
    (wp : writes s b c p loc) (wq : writes s b c q loc) : p=q := by
  exact grouped_injective (s.rows c) (s.cols c) 8 p q (by decide) hp hq
    (wp.2.2.1.trans wq.2.2.1.symm) (wp.2.2.2.trans wq.2.2.2.symm)

 theorem programValue_reads (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (entry current : Memory α)
    (same : ∀ p,b.reads s p → current p=entry p) :
    programValue a s b c mode name current=programValue a s b c mode name entry := by
  funext pid p
  simp only [programValue,sourceAccumulation_eq,preactivation_reads a s b c mode entry current same]

 theorem run_frozen (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (separate : Separate s b) (entry : Memory α) :
    run a s b c mode name entry=
      Schedule.run (Schedule.put (writes s b c) (programValue a s b c mode name entry))
        (List.range (s.programs c)) entry := by
  apply Schedule.freeze_reads (reads:=fun _ => b.reads s) (writes:=writes s b c) (entry:=entry)
  · intro p current same
    unfold programStep
    rw [programValue_reads a s b c mode name entry current same]
  · intro p current loc outside
    exact if_neg outside
  · exact List.nodup_range
  · intro p hp q hq hne loc read write
    exact separate.1 loc read write.1
  · intro p hp loc hread; rfl

 theorem run_cell (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (separate : Separate s b) (entry : Memory α)
    (m n storage : Nat) (hm : m<s.m) (hn : n<s.n)
    (hw : storage=b.out ∨ b.saved=some storage) :
    run a s b c mode name entry (storage,outputOffset s m n)=
      a.encode mode (if storage=b.out then activation a name (preactivation a s b c mode entry m n)
        else preactivation a s b c mode entry m n) := by
  obtain ⟨p,hp,i,j,hi,hj,hr,hc⟩ := source_cell_covered s c m n hm hn
  have active : writes s b c p (storage,outputOffset s m n) := by
    rw [←hr,←hc]
    exact source_store_ownership s b c p i j hi hj (by simpa [hr]) (by simpa [hc]) storage hw
  rw [run_frozen a s b c mode name separate entry,
    Schedule.fixed_output (writes s b c) (programValue a s b c mode name entry)
      (List.range (s.programs c)) entry p _ active]
  · have coords := output_coordinates s m n hn
    simp only [List.mem_range,hp,if_pos,programValue,Prod.fst,Prod.snd,coords.1,coords.2,sourceAccumulation_eq]
  · intro q hq wq
    exact writes_unique s b c q p (by simpa using hq) hp _ wq active

 theorem run_output (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (sep : Separate s b) (entry : Memory α)
    (m n : Nat) (hm : m<s.m) (hn : n<s.n) :
    run a s b c mode name entry (b.out,outputOffset s m n)=
      a.encode mode (activation a name (preactivation a s b c mode entry m n)) := by
  simpa only [ite_true] using run_cell a s b c mode name sep entry m n b.out hm hn (Or.inl rfl)

 theorem run_saved (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (sep : Separate s b) (entry : Memory α)
    (m n saved : Nat) (hm : m<s.m) (hn : n<s.n) (hs : b.saved=some saved) :
    run a s b c mode name entry (saved,outputOffset s m n)=
      a.encode mode (preactivation a s b c mode entry m n) := by
  have hne : saved≠b.out := by intro he; exact sep.2 (he ▸ hs)
  simpa only [hne,ite_false] using run_cell a s b c mode name sep entry m n saved hm hn (Or.inr hs)

 theorem run_frame (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (sep : Separate s b) (entry : Memory α)
    (p : Address) (outside : ¬(b.writesStorage p ∧ p.2<s.m*s.n)) :
    run a s b c mode name entry p=entry p := by
  rw [run_frozen a s b c mode name sep entry]
  apply Schedule.fixed_frame
  intro pid hp hw
  exact outside ⟨hw.1,hw.2.1⟩

 theorem run_empty (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (name : Option String) (entry : Memory α) (empty : s.m=0 ∨ s.n=0) :
    run a s b c mode name entry=entry := by
  have bm : 0<(configs c).bm := by have := config_bounds c; omega
  have bn : 0<(configs c).bn := by have := config_bounds c; omega
  rcases empty with hm|hn
  · simp [run,Schedule.run,Shape.programs,Shape.rows,hm,ceil_zero _ bm]
  · simp [run,Schedule.run,Shape.programs,Shape.cols,hn,ceil_zero _ bn]

#axiomsClean source_store_ownership
#axiomsClean source_store_unique
#axiomsClean writes_unique
#axiomsClean run_frozen
#axiomsClean run_output
#axiomsClean run_saved
#axiomsClean run_frame
#axiomsClean run_empty
end
end VeriTile.Bench.Repairs.TritonLinearActivation
