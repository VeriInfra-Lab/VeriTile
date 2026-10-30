import bench.tritonbench_g.rotary_emb.Proofs.Grid

namespace VeriTile.Bench.Repairs.RotaryEmb
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

def vectorWrites (s : Shape) (p : Ops α) (ws : List Work) (m : Memory α) : List (Stores.Write α) :=
  let output := fun (odd : Bool) (w : Work) =>
    ((dataRegion w.key,s.cell w odd),component (pairValues s p w m) odd)
  ws.map (output false) ++ ws.map (output true)

/-- A source branch: all six loads and both output expressions precede the
even scatter; the odd scatter uses those same loaded values. -/
def vectorBranch (s : Shape) (p : Ops α) (ws : List Work) (m : Memory α) : Memory α :=
  Stores.commit (vectorWrites s p ws m) m

theorem vectorWrites_mem (s : Shape) (p : Ops α) (ws : List Work) (m : Memory α)
    (x : Stores.Write α) : x∈vectorWrites s p ws m ↔
      ∃ w∈ws,∃ odd,x=((dataRegion w.key,s.cell w odd),component (pairValues s p w m) odd) := by
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

theorem vector_coherent (s : Shape) (valid : s.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,s.Active w) (m : Memory α) : Stores.Coherent (vectorWrites s p ws m) := by
  intro a ha b hb equal
  rcases (vectorWrites_mem s p ws m a).mp ha with ⟨w,hw,ow,rfl⟩
  rcases (vectorWrites_mem s p ws m b).mp hb with ⟨v,hv,ov,rfl⟩
  have e := Prod.mk.inj equal
  obtain ⟨rfl,rfl⟩ := cell_injective s valid w v ow ov (active w hw) (active v hv) e.1 e.2
  rfl

theorem vector_readback (s : Shape) (valid : s.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,s.Active w) (m : Memory α) (w : Work) (hw : w∈ws) (odd : Bool) :
    vectorBranch s p ws m (dataRegion w.key) (s.cell w odd)=component (pairValues s p w m) odd := by
  exact Stores.readback _ m (vector_coherent s valid p ws active m) _
    ((vectorWrites_mem s p ws m _).mpr ⟨w,hw,odd,rfl⟩)

theorem vector_frame (s : Shape) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes s w r a) : vectorBranch s p ws m r a=m r a := by
  apply Stores.frame
  intro x hx eq
  rcases (vectorWrites_mem s p ws m x).mp hx with ⟨w,hw,odd,rfl⟩
  have e := Prod.mk.inj eq
  apply outside w hw
  exact ⟨e.1,by cases odd <;> simp_all⟩

theorem vector_eq_execute (s : Shape) (valid : s.Valid) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,s.Active w) (unique : ws.Nodup) (m : Memory α) :
    vectorBranch s p ws m=execute s p ws m := by
  classical
  funext r a
  by_cases touched : ∃ w∈ws,Writes s w r a
  · rcases touched with ⟨w,hw,hr,ha|ha⟩
    · subst r a
      rw [vector_readback s valid p ws active m w hw false,
        execute_readback s valid p ws m active unique w hw false]
    · subst r a
      rw [vector_readback s valid p ws active m w hw true,
        execute_readback s valid p ws m active unique w hw true]
  · have outside : ∀ w∈ws,¬Writes s w r a := by simpa using touched
    rw [vector_frame s p ws m r a outside,execute_frame s p ws m r a outside]

def branchWork (s : Shape) (pid : Program) (key : Bool) : List Work :=
  (((List.range 16) ×ˢ ((List.range 4) ×ˢ (List.range s.pairBlock))).map
    (fun l => workOf (pid,(key,l)))).filter (fun w => decide (s.Active w))

def programWork (s : Shape) (pid : Program) : List Work :=
  branchWork s pid false ++ branchWork s pid true

/-- Q stores precede the K loads and stores, as in the Python source. -/
def sourceProgram (s : Shape) (p : Ops α) (pid : Program) (m : Memory α) : Memory α :=
  let afterQ := vectorBranch s p (branchWork s pid false) m
  vectorBranch s p (branchWork s pid true) afterQ

def sourcePrograms (s : Shape) (p : Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::rest,m => sourcePrograms s p rest (sourceProgram s p pid m)

def source (s : Shape) (p : Ops α) (m : Memory α) := sourcePrograms s p (programs s) m

theorem schedule_flat (s : Shape) : schedule s=(programs s).flatMap (programWork s) := by
  unfold schedule launches
  simp only [SProd.sprod,List.product,List.map_flatMap,List.filter_flatMap]
  apply congrArg (fun f => (programs s).flatMap f)
  funext pid
  simp [lanes,programWork,branchWork,SProd.sprod,List.product,List.map_flatMap,
    List.filter_flatMap,List.map_append,List.filter_append,List.map_map,Function.comp_def]

theorem branch_sub (s : Shape) (pid : Program) (hp : pid∈programs s) (key : Bool)
    (w : Work) (hw : w∈branchWork s pid key) : w∈schedule s := by
  rw [schedule_flat]
  apply List.mem_flatMap.mpr
  refine ⟨pid,hp,?_⟩
  cases key <;> simp [programWork,hw]

theorem list_block_sublist {β γ : Type} (f : β → List γ) (a : β) (xs : List β)
    (ha : a∈xs) : List.Sublist (f a) (xs.flatMap f) := by
  induction xs with
  | nil => simp at ha
  | cons x xs ih =>
    rcases List.mem_cons.mp ha with same|tail
    · subst a
      exact List.sublist_append_left _ _
    · exact (ih tail).trans (List.sublist_append_right _ _)

theorem branch_nodup (s : Shape) (pid : Program) (hp : pid∈programs s) (key : Bool) :
    (branchWork s pid key).Nodup := by
  have sub : List.Sublist (branchWork s pid key) (schedule s) := by
    rw [schedule_flat]
    have hprog : List.Sublist (programWork s pid) ((programs s).flatMap (programWork s)) :=
      list_block_sublist (programWork s) pid (programs s) hp
    cases key
    · exact (List.sublist_append_left _ _).trans hprog
    · exact (List.sublist_append_right _ _).trans hprog
  exact (schedule_nodup s).sublist sub

theorem execute_append (s : Shape) (p : Ops α) (a b : List Work) (m : Memory α) :
    execute s p (a++b) m=execute s p b (execute s p a m) := by
  induction a generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih (step s p w m)

theorem sourceProgram_eq (s : Shape) (valid : s.Valid) (p : Ops α)
    (pid : Program) (hp : pid∈programs s) (m : Memory α) :
    sourceProgram s p pid m=execute s p (programWork s pid) m := by
  unfold sourceProgram
  rw [vector_eq_execute s valid p _
    (fun w hw => schedule_active s w (branch_sub s pid hp false w hw)) (branch_nodup s pid hp false)]
  rw [vector_eq_execute s valid p _
    (fun w hw => schedule_active s w (branch_sub s pid hp true w hw)) (branch_nodup s pid hp true)]
  exact (execute_append s p _ _ m).symm

theorem sourcePrograms_eq (s : Shape) (valid : s.Valid) (p : Ops α) (ps : List Program)
    (inside : ∀ pid∈ps,pid∈programs s) (m : Memory α) :
    sourcePrograms s p ps m=execute s p (ps.flatMap (programWork s)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [sourcePrograms,List.flatMap_cons,execute_append]
    rw [sourceProgram_eq s valid p pid (inside pid (by simp)) m]
    exact ih (fun p hp => inside p (by simp [hp])) _

/-- Constructive source/grid correspondence. No output equation is a premise. -/
theorem source_eq_run (s : Shape) (valid : s.Valid) (p : Ops α) (m : Memory α) :
    source s p m=run s p m := by
  rw [source,sourcePrograms_eq s valid p _ (fun _ h => h),←schedule_flat]
  rfl

#axiomsClean vector_eq_execute
#axiomsClean sourceProgram_eq
#axiomsClean source_eq_run
end VeriTile.Bench.Repairs.RotaryEmb
