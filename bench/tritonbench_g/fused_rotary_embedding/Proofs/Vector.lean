import bench.tritonbench_g.fused_rotary_embedding.Proofs.FusedRotaryEmbedding_Grid

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding
variable {α : Type}

def vectorWrites (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α) : List (Stores.Write α) :=
  let out := fun (o : Bool) (w : Work) =>
    ((w.kind.output,c.output w o),component (pairValues c.shape p w m) o)
  ws.map (out false) ++ ws.map (out true)
def vectorBranch (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α) :=
  Stores.commit (vectorWrites c p ws m) m

theorem vectorWrites_mem (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α)
    (x : Stores.Write α) : x∈vectorWrites c p ws m ↔
      ∃ w∈ws,∃ o,x=((w.kind.output,c.output w o),component (pairValues c.shape p w m) o) := by
  constructor
  · intro h
    rcases List.mem_append.mp h with h|h
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩;exact ⟨w,hw,false,rfl⟩
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩;exact ⟨w,hw,true,rfl⟩
  · rintro ⟨w,hw,o,rfl⟩
    change _∈_++_
    cases o
    · exact List.mem_append_left _ (List.mem_map.mpr ⟨w,hw,rfl⟩)
    · exact List.mem_append_right _ (List.mem_map.mpr ⟨w,hw,rfl⟩)

theorem vector_coherent (c : Config) (hc : c.Valid) (p : Kind → Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.shape.Active w) (m : Memory α) : Stores.Coherent (vectorWrites c p ws m) := by
  intro a ha b hb eq
  rcases (vectorWrites_mem c p ws m a).mp ha with ⟨w,hw,ow,rfl⟩
  rcases (vectorWrites_mem c p ws m b).mp hb with ⟨v,hv,ov,rfl⟩
  have e := Prod.mk.inj eq
  obtain ⟨rfl,rfl⟩ := output_injective c hc w v ow ov (active w hw) (active v hv) e.1 e.2
  rfl
theorem vector_readback (c : Config) (hc : c.Valid) (p : Kind → Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.shape.Active w) (m : Memory α) (w : Work) (hw : w∈ws) (o : Bool) :
    vectorBranch c p ws m w.kind.output (c.output w o)=component (pairValues c.shape p w m) o :=
  Stores.readback _ m (vector_coherent c hc p ws active m) _
    ((vectorWrites_mem c p ws m _).mpr ⟨w,hw,o,rfl⟩)
theorem vector_frame (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes c w r a) : vectorBranch c p ws m r a=m r a := by
  apply Stores.frame
  intro x hx eq
  rcases (vectorWrites_mem c p ws m x).mp hx with ⟨w,hw,o,rfl⟩
  have e := Prod.mk.inj eq
  apply outside w hw
  exact ⟨e.1,by cases o <;> simp_all⟩

theorem vector_eq_execute (c : Config) (hc : c.Valid) (p : Kind → Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.shape.Active w) (unique : ws.Nodup) (m : Memory α) :
    vectorBranch c p ws m=execute c p ws m := by
  classical
  funext r a
  by_cases touched : ∃ w∈ws,Writes c w r a
  · rcases touched with ⟨w,hw,hr,ha|ha⟩
    · subst r a
      rw [vector_readback c hc p ws active m w hw false,execute_readback c hc p ws m active unique w hw false]
    · subst r a
      rw [vector_readback c hc p ws active m w hw true,execute_readback c hc p ws m active unique w hw true]
  · have outside : ∀ w∈ws,¬Writes c w r a := by simpa using touched
    rw [vector_frame c p ws m r a outside,execute_frame c p ws m r a outside]

def branchWork (s : Shape) (pid : Program) (kind : Kind) : List Work :=
  ((((List.range s.halfBlock).map (fun i => (pid,(kind,i)))).filter
    (fun l => decide (leader s l))).map (workOf s)).filter (fun w => decide (s.Active w))
def programWork (s : Shape) (pid : Program) :=
  branchWork s pid .q ++ branchWork s pid .k ++ branchWork s pid .v

theorem schedule_flat (s : Shape) : schedule s=(programs s).flatMap (programWork s) := by
  unfold schedule selectedLanes launches
  simp only [SProd.sprod,List.product,List.filter_flatMap,List.map_flatMap]
  apply congrArg (fun f => (programs s).flatMap f)
  funext pid
  simp [lanes,programWork,branchWork,SProd.sprod,List.product,List.map_flatMap,
    List.filter_flatMap,List.map_append,List.filter_append,List.map_map,Function.comp_def]

theorem branch_sub (s : Shape) (pid : Program) (hp : pid∈programs s) (kind : Kind)
    (w : Work) (hw : w∈branchWork s pid kind) : w∈schedule s := by
  rw [schedule_flat]
  apply List.mem_flatMap.mpr
  refine ⟨pid,hp,?_⟩
  cases kind <;> simp [programWork,hw]

theorem list_block_sublist {β γ : Type} (f : β → List γ) (a : β) (xs : List β)
    (ha : a∈xs) : List.Sublist (f a) (xs.flatMap f) := by
  induction xs with
  | nil => simp at ha
  | cons x xs ih =>
    rcases List.mem_cons.mp ha with same|tail
    · subst a;exact List.sublist_append_left _ _
    · exact (ih tail).trans (List.sublist_append_right _ _)

theorem branch_nodup (s : Shape) (pid : Program) (hp : pid∈programs s) (kind : Kind) :
    (branchWork s pid kind).Nodup := by
  have sub : List.Sublist (branchWork s pid kind) (programWork s pid) := by
    cases kind
    · exact (List.sublist_append_left _ _).trans (List.sublist_append_left _ _)
    · exact (List.sublist_append_right _ _).trans (List.sublist_append_left _ _)
    · exact List.sublist_append_right _ _
  have all := list_block_sublist (programWork s) pid (programs s) hp
  rw [←schedule_flat] at all
  exact (schedule_nodup s).sublist (sub.trans all)

/-- The Q vector stores precede K operand loads and stores, which precede V
loads and stores. Nonleader K/V branches are empty by the actual leader guard. -/
def sourceProgram (c : Config) (p : Kind → Ops α) (pid : Program) (m : Memory α) : Memory α :=
  let afterQ := vectorBranch c p (branchWork c.shape pid .q) m
  let afterK := vectorBranch c p (branchWork c.shape pid .k) afterQ
  vectorBranch c p (branchWork c.shape pid .v) afterK
def sourcePrograms (c : Config) (p : Kind → Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::ps,m => sourcePrograms c p ps (sourceProgram c p pid m)
def source (c : Config) (p : Kind → Ops α) (m : Memory α) := sourcePrograms c p (programs c.shape) m

theorem sourceProgram_eq (c : Config) (hc : c.Valid) (p : Kind → Ops α)
    (pid : Program) (hp : pid∈programs c.shape) (m : Memory α) :
    sourceProgram c p pid m=execute c p (programWork c.shape pid) m := by
  unfold sourceProgram
  dsimp only
  rw [vector_eq_execute c hc p _
    (fun w hw => schedule_active _ w (branch_sub _ pid hp .q w hw)) (branch_nodup _ pid hp .q)]
  rw [vector_eq_execute c hc p _
    (fun w hw => schedule_active _ w (branch_sub _ pid hp .k w hw)) (branch_nodup _ pid hp .k)]
  rw [vector_eq_execute c hc p _
    (fun w hw => schedule_active _ w (branch_sub _ pid hp .v w hw)) (branch_nodup _ pid hp .v)]
  simp only [programWork,execute_append]

theorem sourcePrograms_eq (c : Config) (hc : c.Valid) (p : Kind → Ops α) (ps : List Program)
    (inside : ∀ pid∈ps,pid∈programs c.shape) (m : Memory α) :
    sourcePrograms c p ps m=execute c p (ps.flatMap (programWork c.shape)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    simp only [sourcePrograms,List.flatMap_cons,execute_append]
    rw [sourceProgram_eq c hc p pid (inside pid (by simp)) m]
    exact ih (fun pid hp => inside pid (by simp [hp])) _
theorem source_eq_run (c : Config) (hc : c.Valid) (p : Kind → Ops α) (m : Memory α) :
    source c p m=run c p m := by
  rw [source,sourcePrograms_eq c hc p _ (fun _ h => h),←schedule_flat]
  rfl

/-- Reusing the coefficient vectors loaded before Q stores is exact. -/
theorem coefficient_reuse (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α)
    (sine : Bool) (a : Nat) :
    vectorBranch c p ws m (if sine then .sin else .cos) a=m (if sine then .sin else .cos) a := by
  apply vector_frame
  intro w _
  cases sine <;> cases kw : w.kind <;> simp [Writes,Kind.output,kw]

#axiomsClean vector_eq_execute
#axiomsClean source_eq_run
#axiomsClean coefficient_reuse
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
