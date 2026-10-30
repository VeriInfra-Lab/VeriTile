import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Whole
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Source

/-- Initial-state stores share the reverse kernel's key/value/head grid. -/
def eventsInitial (c : Shape) : List Address :=
  programs c.keyBlocks c.valueBlocks c.heads fun kb vb h =>
    tile .initialGradient h 0 (kb*c.BK) (vb*c.BV) c.BK c.BV c.K c.V

theorem initial_footprint (c : Shape) (a : Address) (ha : a∈eventsInitial c) :
    a.region=.initialGradient := by
  simp only [eventsInitial,programs,List.mem_flatMap,List.mem_range] at ha
  rcases ha with ⟨h,hh,vb,hv,kb,hk,ha⟩
  exact tile_tag _ _ _ _ _ _ _ _ _ a ha

theorem initial_cover (c : Shape) (h e j : Nat) (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    (⟨.initialGradient,h,0,e,j⟩:Address)∈eventsInitial c := by
  apply program_member _ _ _ _ (e/c.BK) (j/c.BV) h _
    (Shape.feature_block _ _ _ c.key_tile_pos he) (Shape.feature_block _ _ _ c.value_tile_pos hj) hh
  have hm := tile_member .initialGradient h 0 (e/c.BK*c.BK) (j/c.BV*c.BV) c.BK c.BV c.K c.V
    (e%c.BK) (j%c.BV) (Nat.mod_lt e c.key_tile_pos) (Nat.mod_lt j c.value_tile_pos)
  simpa [Nat.div_add_mod'] using hm (by simpa [Nat.div_add_mod'] using he) (by simpa [Nat.div_add_mod'] using hj)

def ExceptInitial (a b : Heap α) : Prop := ∀ p, p.region≠.initialGradient → a p=b p

theorem project_except_initial (ops : Ops α) (p : Phase) (a b : Heap α)
    (same : ExceptInitial a b) : project ops p a=project ops p b := by
  funext point
  unfold project
  split_ifs with hr
  · apply same
    intro he
    rw [he] at hr
    cases p <;> simp [Phase.reads,inputRegions] at hr
  · rfl

theorem rhs_except_initial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (p : Phase) (a b : Heap α) (same : ExceptInitial a b) :
    rhs ops c scale tree p a=rhs ops c scale tree p b := by
  unfold rhs
  rw [project_except_initial ops p a b same]

theorem execute_except_initial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (p : Phase) (events : List Address) (a b : Heap α) (same : ExceptInitial a b) :
    ExceptInitial (execute ops c scale tree p events a) (execute ops c scale tree p events b) := by
  induction events generalizing a b with
  | nil => exact same
  | cons point rest ih =>
    apply ih
    intro other ho
    unfold Heap.write
    rw [rhs_except_initial ops c scale tree p a b same]
    split_ifs
    · rfl
    · exact same other ho

theorem execute_append (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (p : Phase) (left right : List Address) (entry : Heap α) :
    execute ops c scale tree p (left++right) entry=
      execute ops c scale tree p right (execute ops c scale tree p left entry) := by
  induction left generalizing entry with
  | nil => rfl
  | cons a rest ih => simp only [List.cons_append,execute,ih]

/-- The optional stores are appended within the same reverse phase. All stores
in that phase read inputs only, so execute_spec proves invariance under any
permutation of its event list, including the native per-program interleaving. -/
def afterDHInitial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (returnFinal returnInitial : Bool) (entry : Heap α) :=
  execute ops c scale tree .dh
    (eventsDH c ++ if returnInitial then eventsInitial c else [])
    (afterO ops c scale tree returnFinal entry)

def runAPI (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (returnFinal returnInitial : Bool) (entry : Heap α) :=
  execute ops c scale tree .sum (eventsSum c)
    (execute ops c scale tree .qkv (eventsQKV c)
      (afterDHInitial ops c scale tree returnFinal returnInitial entry))

theorem initial_events_writes (c : Shape) (flag : Bool) (a : Address)
    (ha : a∈eventsDH c ++ if flag then eventsInitial c else []) : a.region∈Phase.dh.writes := by
  rcases List.mem_append.mp ha with ha|ha
  · exact dh_footprint c a ha
  · cases flag <;> simp only [Bool.false_eq_true,↓reduceIte,List.not_mem_nil] at ha
    rw [initial_footprint c a ha]
    simp [Phase.writes]

theorem afterDH_initial_agrees (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf ri : Bool) (entry : Heap α) :
    ExceptInitial (afterDHInitial ops c scale tree rf ri entry) (afterDH ops c scale tree rf entry) := by
  intro point hp
  rw [afterDHInitial,execute_append]
  cases ri
  · rfl
  ·
    rw [execute_spec ops c scale tree .dh _ (fun a ha => by
      rw [initial_footprint c a ha]; simp [Phase.writes])]
    simp only [↓reduceIte]
    rw [if_neg (fun ha => hp (initial_footprint c point ha))]
    rfl

theorem runAPI_agrees (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf ri : Bool) (entry : Heap α) :
    ExceptInitial (runAPI ops c scale tree rf ri entry) (run ops c scale tree rf entry) := by
  apply execute_except_initial
  apply execute_except_initial
  exact afterDH_initial_agrees ops c scale tree rf ri entry

theorem runAPI_initial (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf : Bool) (entry : Heap α) (h e j : Nat)
    (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    runAPI ops c scale tree rf Bool.true entry ⟨.initialGradient,h,0,e,j⟩=
      ops.cast (Reference.reverseCarry ops c (inputsOfHeap entry) scale h c.chunks e j) := by
  rw [runAPI,execute_frame ops c scale tree .sum _ (sum_footprint c) _ _ (by simp [Phase.writes]),
    execute_frame ops c scale tree .qkv _ (qkv_footprint c) _ _ (by simp [Phase.writes]),
    afterDHInitial,execute_spec ops c scale tree .dh _ (initial_events_writes c Bool.true)]
  have hm : (⟨.initialGradient,h,0,e,j⟩:Address)∈eventsDH c ++ if Bool.true then eventsInitial c else [] := by
    apply List.mem_append_right
    exact initial_cover c h e j hh he hj
  rw [if_pos hm]
  simp only [rhs,bodyRHS,project_inputs ops .dh (by decide),o_inputs,reduceCtorEq,↓reduceIte]

theorem runAPI_frame (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (rf ri : Bool) (entry : Heap α) (point : Address)
    (outside : point∉footprint c rf)
    (initialOutside : point∉if ri then eventsInitial c else []) :
    runAPI ops c scale tree rf ri entry point=entry point := by
  simp only [footprint,List.mem_append,not_or] at outside
  rw [runAPI,execute_spec ops c scale tree .sum _ (sum_footprint c),if_neg outside.2,
    execute_spec ops c scale tree .qkv _ (qkv_footprint c),if_neg outside.1.2,
    afterDHInitial,execute_spec ops c scale tree .dh _ (initial_events_writes c ri),
    if_neg (by simp only [List.mem_append]; exact not_or.mpr ⟨outside.1.1.2,initialOutside⟩),
    afterO,execute_spec ops c scale tree .o _ (o_footprint c),if_neg outside.1.1.1.2,
    afterH,execute_spec ops c scale tree .h _ (h_footprint c rf),if_neg outside.1.1.1.1]

#axiomsClean runAPI_initial
#axiomsClean runAPI_agrees
#axiomsClean runAPI_frame
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

