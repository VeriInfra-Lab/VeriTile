import bench.repairs.tritonbench_g.rotary_transform.Proofs.Grid

namespace VeriTile.Bench.Repairs.RotaryTransform
namespace Stores
variable {α : Type}
abbrev Write (α : Type) := (Region × Nat) × α

def store (m : Memory α) (w : Write α) : Memory α :=
  fun r a => if (r,a)=w.1 then w.2 else m r a

def commit (ws : List (Write α)) (m : Memory α) : Memory α := ws.foldl store m

def Coherent (ws : List (Write α)) : Prop :=
  ∀ a∈ws,∀ b∈ws,a.1=b.1 → a.2=b.2

theorem frame (ws : List (Write α)) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w∈ws,(r,a)≠w.1) : commit ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change commit ws (store m w) r a=m r a
    rw [ih _ (fun x hx => outside x (by simp [hx]))]
    simp [store,outside w (by simp)]

theorem stable (ws : List (Write α)) (m : Memory α) (r : Region) (a : Nat) (value : α)
    (old : m r a=value) (agree : ∀ w∈ws,w.1=(r,a) → w.2=value) :
    commit ws m r a=value := by
  induction ws generalizing m with
  | nil => exact old
  | cons w ws ih =>
    apply ih (store m w)
    · by_cases eq : (r,a)=w.1
      · simp [store,eq,agree w (by simp) eq.symm]
      · simp [store,eq,old]
    · exact fun x hx eq => agree x (by simp [hx]) eq

theorem readback (ws : List (Write α)) (m : Memory α) (good : Coherent ws)
    (w : Write α) (member : w∈ws) : commit ws m w.1.1 w.1.2=w.2 := by
  induction ws generalizing m with
  | nil => simp at member
  | cons x ws ih =>
    change commit ws (store m x) w.1.1 w.1.2=w.2
    rcases List.mem_cons.mp member with eq|tail
    · subst w
      apply stable
      · simp [store]
      · intro v hv same
        exact good v (by simp [hv]) x (by simp) same
    · exact ih _ (fun a ha b hb eq => good a (by simp [ha]) b (by simp [hb]) eq) tail
end Stores

variable {α : Type}

def vectorWrites (c : Config) (p : Ops α) (ws : List Work) (m : Memory α) : List (Stores.Write α) :=
  let output := fun (odd : Bool) (w : Work) =>
    ((c.shape.output,c.cell w odd),component (pairValues c p w m) odd)
  ws.map (output false) ++ ws.map (output true)

/-- A source branch: all six loads and both output expressions precede the
even scatter; the odd scatter uses those same loaded values. -/
def vectorBranch (c : Config) (p : Ops α) (ws : List Work) (m : Memory α) : Memory α :=
  Stores.commit (vectorWrites c p ws m) m

theorem vectorWrites_mem (c : Config) (p : Ops α) (ws : List Work) (m : Memory α)
    (x : Stores.Write α) : x∈vectorWrites c p ws m ↔
      ∃ w∈ws,∃ odd,x=((c.shape.output,c.cell w odd),component (pairValues c p w m) odd) := by
  constructor
  · intro h
    rcases List.mem_append.mp h with h|h
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩
      exact ⟨w,hw,false,rfl⟩
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩
      exact ⟨w,hw,true,rfl⟩
  · rintro ⟨w,hw,odd,rfl⟩
    change _∈_++_
    cases odd
    · exact List.mem_append_left _ (List.mem_map.mpr ⟨w,hw,rfl⟩)
    · exact List.mem_append_right _ (List.mem_map.mpr ⟨w,hw,rfl⟩)

theorem vector_coherent (c : Config) (valid : c.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.Active w) (m : Memory α) : Stores.Coherent (vectorWrites c p ws m) := by
  intro a ha b hb equal
  rcases (vectorWrites_mem c p ws m a).mp ha with ⟨w,hw,ow,rfl⟩
  rcases (vectorWrites_mem c p ws m b).mp hb with ⟨v,hv,ov,rfl⟩
  have e := Prod.mk.inj equal
  obtain ⟨rfl,rfl⟩ := cell_injective c valid w v ow ov (active w hw) (active v hv) e.2
  rfl

theorem vector_readback (c : Config) (valid : c.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.Active w) (m : Memory α) (w : Work) (hw : w∈ws) (odd : Bool) :
    vectorBranch c p ws m (c.shape.output) (c.cell w odd)=component (pairValues c p w m) odd := by
  exact Stores.readback _ m (vector_coherent c valid p ws active m) _
    ((vectorWrites_mem c p ws m _).mpr ⟨w,hw,odd,rfl⟩)

theorem vector_frame (c : Config) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes c w r a) : vectorBranch c p ws m r a=m r a := by
  apply Stores.frame
  intro x hx eq
  rcases (vectorWrites_mem c p ws m x).mp hx with ⟨w,hw,odd,rfl⟩
  have e := Prod.mk.inj eq
  apply outside w hw
  exact ⟨e.1,by cases odd <;> simp_all⟩

theorem vector_eq_execute (c : Config) (valid : c.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.Active w) (unique : ws.Nodup) (m : Memory α) :
    vectorBranch c p ws m=execute c p ws m := by
  classical
  funext r a
  by_cases touched : ∃ w∈ws,Writes c w r a
  · rcases touched with ⟨w,hw,hr,ha|ha⟩
    · subst r a
      rw [vector_readback c valid p ws active m w hw false,
        execute_readback c valid p ws m active unique w hw false]
    · subst r a
      rw [vector_readback c valid p ws active m w hw true,
        execute_readback c valid p ws m active unique w hw true]
  · have outside : ∀ w∈ws,¬Writes c w r a := by simpa using touched
    rw [vector_frame c p ws m r a outside,execute_frame c p ws m r a outside]

def programWork (c : Config) (pid : Program) : List Work :=
  ((lanes c).map (fun l => workOf c (pid,l))).filter (fun w => decide (c.Active w))
def sourceProgram (c : Config) (p : Ops α) (pid : Program) (m : Memory α) := vectorBranch c p (programWork c pid) m
def sourcePrograms (c : Config) (p : Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::rest,m => sourcePrograms c p rest (sourceProgram c p pid m)
def source (c : Config) (p : Ops α) (m : Memory α) := sourcePrograms c p (programs c) m
theorem schedule_flat (c : Config) : schedule c=(programs c).flatMap (programWork c) := by
  simp [schedule,launches,programWork,SProd.sprod,List.product,List.map_flatMap,List.filter_flatMap,List.map_map,Function.comp_def]
  rfl
theorem program_sub (c : Config) (pid : Program) (hp : pid∈programs c)
    (w : Work) (hw : w∈programWork c pid) : w∈schedule c := by
  rw [schedule_flat]
  exact List.mem_flatMap.mpr ⟨pid,hp,hw⟩
theorem list_block_sublist {β γ : Type} (f : β→List γ) (a : β) (xs : List β)
    (ha : a∈xs) : List.Sublist (f a) (xs.flatMap f) := by
  induction xs with
  | nil => simp at ha
  | cons x xs ih =>
    rcases List.mem_cons.mp ha with same|tail
    · subst a;exact List.sublist_append_left _ _
    · exact (ih tail).trans (List.sublist_append_right _ _)
theorem program_nodup (c : Config) (pid : Program) (hp : pid∈programs c) : (programWork c pid).Nodup := by
  have sub : List.Sublist (programWork c pid) (schedule c) := by
    rw [schedule_flat]
    exact list_block_sublist (programWork c) pid (programs c) hp
  exact (schedule_nodup c).sublist sub
theorem sourceProgram_eq (c : Config) (hc : c.Valid) (p : Ops α) (pid : Program)
    (hp : pid∈programs c) (m : Memory α) : sourceProgram c p pid m=execute c p (programWork c pid) m :=
  vector_eq_execute c hc p _ (fun w hw => schedule_active c w (program_sub c pid hp w hw)) (program_nodup c pid hp) m
theorem sourcePrograms_eq (c : Config) (hc : c.Valid) (p : Ops α) (ps : List Program)
    (inside : ∀ pid∈ps,pid∈programs c) (m : Memory α) :
    sourcePrograms c p ps m=execute c p (ps.flatMap (programWork c)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [sourcePrograms,List.flatMap_cons,execute_append]
    rw [sourceProgram_eq c hc p pid (inside pid (by simp))]
    exact ih (fun p hp => inside p (by simp [hp])) _
theorem source_eq_run (c : Config) (hc : c.Valid) (p : Ops α) (m : Memory α) : source c p m=run c p m := by
  rw [source,sourcePrograms_eq c hc p _ (fun _ h => h),←schedule_flat];rfl

#axiomsClean vector_eq_execute
#axiomsClean sourceProgram_eq
#axiomsClean source_eq_run
end VeriTile.Bench.Repairs.RotaryTransform
