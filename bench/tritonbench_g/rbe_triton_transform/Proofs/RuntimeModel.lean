import bench.tritonbench_g.rbe_triton_transform.Proofs.Bridge

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Runtime
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

abbrev Program := Nat × Nat
def location (q : Program) (i : Lane) (im : Bool) : Location := (q.1,q.2,i,im)
def programs (r : Request) : List Program :=
  (List.range r.batches).flatMap fun b => (List.range (grid r)).map fun p => (b,p)
def setup (s : State α) (q : Program) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun d => if d=0 then q.1 else if d=1 then q.2 else 0}

theorem setup_location (s : State α) (q : Program) (i : Lane) (im : Bool) :
    (setup s q).mem=(programState s (location q i im)).mem := rfl

def block (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program) (s : State α) :=
  Source.written ops.backend (parameters r X Y) (setup s q)

def compose (ops : Operations α) (r : Request) (X Y : RegionName) : List Program → State α → State α
  | [],s => s
  | q::rest,s => compose ops r X Y rest (block ops r X Y q s)

theorem scatter_congr {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Cell α)
    (s t : State α) (h : s.mem=t.mem) :
    (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) s).mem =
      (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

theorem block_congr (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s t : State α) (h : s.mem=t.mem) :
    (block ops r X Y q s).mem=(block ops r X Y q t).mem := by
  have values : ∀ im i, Source.value ops.backend (parameters r X Y) (setup s q) im i=
      Source.value ops.backend (parameters r X Y) (setup t q) im i := by
    intro im i
    simp [Source.value,Source.component,Source.phase,Source.active,Source.inputOffset,
      Source.row,Source.column,setup,h]
  simp only [block,Source.written,Source.halfScatter,values]
  apply scatter_congr
  apply scatter_congr
  exact h

theorem compose_congr (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s t : State α) (h : s.mem=t.mem) :
    (compose ops r X Y qs s).mem=(compose ops r X Y qs t).mem := by
  induction qs generalizing s t with
  | nil => exact h
  | cons q rest ih => exact ih _ _ (block_congr ops r X Y q s t h)

def Hits (r : Request) (q : Program) (o : Nat) : Prop :=
  ∃ i im, active r (location q i im) ∧ o=r.offset (batch (location q i im))
    (row r (location q i im)) (column r (location q i im))

theorem block_frame (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s : State α) (region : RegionName) (o : Nat) (hq : q.1<r.batches)
    (outside : region≠Y ∨ ¬Hits r q o) :
    (block ops r X Y q s).mem region o=s.mem region o := by
  apply Source.written_frame
  rcases outside with different|absent
  · exact .inl different
  · right
    intro im i hi same
    apply absent
    refine ⟨i,im,?_,?_⟩
    · refine ⟨hq,hi.1,?_⟩
      cases im <;> simpa [Source.active,column,Source.column,Source.row,setup,location,
        row,evenColumn,program,lane,imaginary,parameters,Source.parts,parts,Nat.add_comm] using hi.2
    · exact same.trans (source_offset r X Y s (location q i im)).2

theorem block_input (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s : State α) (different : X≠Y) :
    ∀ o,(block ops r X Y q s).mem X o=s.mem X o := by
  intro o
  exact Source.written_frame ops.backend (parameters r X Y) (setup s q) X o (.inl different)

theorem block_value (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s : State α) (i : Lane) (im : Bool) (ha : active r (location q i im)) :
    ((block ops r X Y q s).mem Y (r.offset (batch (location q i im))
      (row r (location q i im)) (column r (location q i im)))).read .fp32=
      scalarResult ops r (fun o => (s.mem X o).read .fp32)
        (batch (location q i im)) (row r (location q i im)) (column r (location q i im)) := by
  have hv := Source.written_readback ops.backend (parameters r X Y) (setup s q)
    (source_apart r X Y (setup s q)) im i (source_active r X Y s (location q i im) ha)
  have off : Source.outputOffset (parameters r X Y) (setup s q) im i=
      r.offset (batch (location q i im)) (row r (location q i im)) (column r (location q i im)) :=
    (source_offset r X Y (setup s q) (location q i im)).2
  rw [off] at hv
  rw [block,hv,Cell.read_same]
  exact source_value ops r X Y s (location q i im) ha

theorem block_result (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s : State α) (x : Nat → α) (inputs : ∀ o,(s.mem X o).read .fp32=x o)
    (b m c : Nat) (hm : m<r.rows) (hc : c<r.columns) (hit : Hits r q (r.offset b m c)) :
    ((block ops r X Y q s).mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c := by
  obtain ⟨i,im,ha,he⟩ := hit
  obtain ⟨hb,hr,hcol⟩ := offset_injective r ha.2.1 ha.2.2 hm hc he.symm
  have value := block_value ops r X Y q s i im ha
  simpa only [hb,hr,hcol,inputs] using value

theorem block_keeps (ops : Operations α) (r : Request) (X Y : RegionName) (q : Program)
    (s : State α) (x : Nat → α) (inputs : ∀ o,(s.mem X o).read .fp32=x o)
    (b m c : Nat) (hm : m<r.rows) (hc : c<r.columns) (hq : q.1<r.batches)
    (before : (s.mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c) :
    ((block ops r X Y q s).mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c := by
  by_cases hit : Hits r q (r.offset b m c)
  · exact block_result ops r X Y q s x inputs b m c hm hc hit
  · rw [block_frame ops r X Y q s Y _ hq (.inr hit)]
    exact before

theorem compose_keeps (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) (x : Nat → α) (inputs : ∀ o,(s.mem X o).read .fp32=x o)
    (b m c : Nat) (hm : m<r.rows) (hc : c<r.columns) (valid : ∀ q∈qs,q.1<r.batches)
    (different : X≠Y) (before : (s.mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c) :
    ((compose ops r X Y qs s).mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c := by
  induction qs generalizing s with
  | nil => exact before
  | cons q rest ih =>
    apply ih
    · intro o
      rw [block_input ops r X Y q s different]
      exact inputs o
    · exact fun p hp => valid p (by simp [hp])
    · exact block_keeps ops r X Y q s x inputs b m c hm hc (valid q (by simp)) before

theorem compose_value (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) (x : Nat → α) (inputs : ∀ o,(s.mem X o).read .fp32=x o)
    (b m c : Nat) (hm : m<r.rows) (hc : c<r.columns) (valid : ∀ q∈qs,q.1<r.batches)
    (different : X≠Y) (covered : ∃ q∈qs,Hits r q (r.offset b m c)) :
    ((compose ops r X Y qs s).mem Y (r.offset b m c)).read .fp32=scalarResult ops r x b m c := by
  induction qs generalizing s with
  | nil => simp at covered
  | cons q rest ih =>
    have inputs' : ∀ o,((block ops r X Y q s).mem X o).read .fp32=x o := by
      intro o
      rw [block_input ops r X Y q s different]
      exact inputs o
    have valid' : ∀ p∈rest,p.1<r.batches := fun p hp => valid p (by simp [hp])
    by_cases hit : Hits r q (r.offset b m c)
    · exact compose_keeps ops r X Y rest _ x inputs' b m c hm hc valid' different
        (block_result ops r X Y q s x inputs b m c hm hc hit)
    · apply ih _ inputs' valid'
      obtain ⟨p,hp,h⟩ := covered
      exact ⟨p,(List.mem_cons.mp hp).resolve_left (by intro he; subst p; exact hit h),h⟩

theorem mem_programs (r : Request) (q : Program) :
    q∈programs r ↔ q.1<r.batches ∧ q.2<grid r := by
  rcases q with ⟨b,p⟩
  simp [programs]

theorem compose_frame (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) (valid : ∀ q∈qs,q.1<r.batches)
    (region : RegionName) (o : Nat) (outside : region≠Y ∨ r.size≤o) :
    (compose ops r X Y qs s).mem region o=s.mem region o := by
  induction qs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    rw [compose,ih _ (fun p hp => valid p (by simp [hp]))]
    apply block_frame ops r X Y q s region o (valid q (by simp))
    rcases outside with different|bound
    · exact .inl different
    · right
      rintro ⟨i,im,ha,he⟩
      have := offset_lt r _ _ _ ha.1 ha.2.1 ha.2.2
      omega

theorem programs_empty (r : Request) (empty : r.size=0) : programs r=[] := by
  have split : r.batches=0 ∨ r.rows=0 ∨ r.columns=0 := by
    simpa [Request.size,Nat.mul_eq_zero,or_assoc] using empty
  rcases split with h|h|h <;> simp [programs,grid,rowParts,parts,h]


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Runtime
