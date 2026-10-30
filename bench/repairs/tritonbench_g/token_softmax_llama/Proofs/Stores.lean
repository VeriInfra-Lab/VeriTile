import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Grid
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

abbrev Event (p : Parameters) := PID × TileIndex [p.width]

def events {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (work : List PID) : List (Event p) :=
  work.flatMap (fun pid => ((TileShape.allIndices [p.width]).filter (fun i => decide (active p base pid i))).map (fun i => (pid,i)))

def write {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (vals : Event p → α) (m : Memory α) (e : Event p) : Memory α :=
  fun r o => if r=p.output ∧ o=offset p base e.1 e.2 then .mk .fp32 (vals e) else m r o

def values {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (base : Memory α) (e : Event p) := outputValues M .fp32 p (state base e.1) e.2

theorem snapshot_events {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (base m : Memory α) (work : List PID) :
    snapshot M p base work m=(events p base work).foldl (write p base (values M p base)) m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [snapshot,events,List.flatMap_cons,List.foldl_append,List.foldl_map,List.foldl_filter]
    rw [ih]
    simp only [Bool.decide_coe,decide_eq_true_eq]
    rfl

theorem mem_events {α : Type} [Inhabited α] (p : Parameters) (base : Memory α)
    (work : List PID) (e : Event p) :
    e∈events p base work ↔ e.1∈work ∧ active p base e.1 e.2 := by
  simp only [events,List.mem_flatMap,List.mem_map,List.mem_filter,TileShape.mem_allIndices,true_and,decide_eq_true_eq]
  constructor
  · rintro ⟨pid,hpid,i,hi,eq⟩
    cases eq
    exact ⟨hpid,hi.2⟩
  · intro h
    exact ⟨e.1,h.1,e.2,⟨TileShape.mem_allIndices _ _,h.2⟩,rfl⟩

theorem writes_frame {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (vals : Event p → α)
    (m : Memory α) (es : List (Event p)) (r : RegionName) (o : Nat)
    (outside : r≠p.output ∨ ∀ e∈es, o≠offset p base e.1 e.2) :
    (es.foldl (write p base vals) m) r o=m r o := by
  induction es generalizing m with
  | nil => rfl
  | cons e es ih =>
    rw [List.foldl_cons,ih]
    · unfold write
      split
      · rename_i h
        rcases outside with hr|ho
        · exact False.elim (hr h.1)
        · exact False.elim (ho e (by simp) h.2)
      · rfl
    · rcases outside with hr|ho
      · exact Or.inl hr
      · exact Or.inr (fun x hx => ho x (List.mem_cons_of_mem _ hx))

theorem writes_preserve {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (vals : Event p → α)
    (m : Memory α) (es : List (Event p)) (target : Event p)
    (initial : m p.output (offset p base target.1 target.2)=.mk .fp32 (vals target))
    (compatible : ∀ e∈es, offset p base e.1 e.2=offset p base target.1 target.2 → vals e=vals target) :
    (es.foldl (write p base vals) m) p.output (offset p base target.1 target.2)=.mk .fp32 (vals target) := by
  induction es generalizing m with
  | nil => exact initial
  | cons e es ih =>
    rw [List.foldl_cons]
    apply ih
    · by_cases same : offset p base target.1 target.2=offset p base e.1 e.2
      · simp [write,same,compatible e (by simp) same.symm]
      · simpa [write,same] using initial
    · intro x hx
      exact compatible x (List.mem_cons_of_mem _ hx)

theorem writes_readback {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (vals : Event p → α)
    (m : Memory α) (es : List (Event p)) (target : Event p) (live : target∈es)
    (compatible : ∀ e∈es, offset p base e.1 e.2=offset p base target.1 target.2 → vals e=vals target) :
    (es.foldl (write p base vals) m) p.output (offset p base target.1 target.2)=.mk .fp32 (vals target) := by
  induction es generalizing m with
  | nil => simp at live
  | cons e es ih =>
    rw [List.foldl_cons]
    rcases List.mem_cons.mp live with same|later
    · subst e
      apply writes_preserve
      · simp [write]
      · intro x hx
        exact compatible x (List.mem_cons_of_mem _ hx)
    · apply ih _ later
      intro x hx
      exact compatible x (List.mem_cons_of_mem _ hx)

def fullGrid (batches heads : Nat) : List PID :=
  (List.range batches).flatMap (fun b => (List.range heads).map (fun h => (b,h)))

theorem mem_fullGrid (batches heads : Nat) (pid : PID) :
    pid∈fullGrid batches heads ↔ pid.1 < batches ∧ pid.2 < heads := by
  rcases pid with ⟨b,h⟩
  simp [fullGrid,Prod.mk.injEq]

def Unique {α : Type} [Inhabited α] (p : Parameters) (base : Memory α) (work : List PID) :=
  ∀ e∈events p base work, ∀ f∈events p base work,
    offset p base e.1 e.2=offset p base f.1 f.2 → e=f

theorem whole {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (positive : 0 < p.width) (work : List PID) (m : Memory α) (separate : Separate p)
    (unique : Unique p m work) :
    ∃ final, run M p work m=some final ∧
      final=(events p m work).foldl (write p m (values M p m)) m ∧
      (∀ e∈events p m work, final p.output (offset p m e.1 e.2)=.mk .fp32 (values M p m e)) ∧
      (∀ r o, (r≠p.output ∨ ∀ e∈events p m work,o≠offset p m e.1 e.2) → final r o=m r o) := by
  have eq := (graph_snapshot M p work m m separate (fun _ _ => rfl)).trans
    (snapshot_events M p m m work)
  refine ⟨graph M p work m,complete M p positive work m,eq,?_,?_⟩
  · intro e he
    rw [eq]
    apply writes_readback p m _ m _ e he
    intro f hf h
    rw [unique f hf e he h]
  · intro r o outside
    rw [eq]
    exact writes_frame p m _ m _ r o outside

#axiomsClean snapshot_events
#axiomsClean mem_events
#axiomsClean mem_fullGrid
#axiomsClean whole
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided.Grid
