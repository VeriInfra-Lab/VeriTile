import bench.tritonbench_g.decay_cumsum.Proofs.Geometry

namespace VeriTile.Bench.Repairs.DecayCumsum.Coverage
open Geometry
set_option maxHeartbeats 4000000

theorem time_owner (h : Call) (t : Fin h.T) :
    ∃ p, p∈phases h ∧ ∃ chunk : Fin p.count, ∃ step : Fin p.span,
      time p chunk.val step.val=t.val := by
  have division := Nat.mod_add_div h.T h.BT
  have divisionT := Nat.mod_add_div t.val h.BT
  have ht := t.isLt
  by_cases main : t.val<h.T/h.BT*h.BT
  · have count : h.T/h.BT≠0 := by intro hz; simp [hz] at main
    have member : mainPhase h∈phases h := by
      simp only [phases,List.mem_append,if_neg count,List.mem_singleton]
      exact Or.inl trivial
    have chunk : t.val/h.BT<h.T/h.BT := (Nat.div_lt_iff_lt_mul h.timeTilePositive).mpr main
    refine ⟨mainPhase h,member,⟨t.val/h.BT,chunk⟩,⟨t.val%h.BT,Nat.mod_lt _ h.timeTilePositive⟩,?_⟩
    dsimp [time,mainPhase]
    nlinarith
  · have count : h.T%h.BT≠0 := by intro hz; nlinarith
    have member : tailPhase h∈phases h := by
      simp only [phases,List.mem_append,if_neg count,List.mem_singleton]
      exact Or.inr trivial
    have step : t.val-h.T/h.BT*h.BT<h.T%h.BT := by
      have total : h.T/h.BT*h.BT+h.T%h.BT=h.T := by nlinarith
      omega
    refine ⟨tailPhase h,member,⟨0,by simp [tailPhase]⟩,⟨t.val-h.T/h.BT*h.BT,step⟩,?_⟩
    dsimp [time,tailPhase]
    omega

theorem coordinate_owner (h : Call) (row : Fin h.rows) (t : Fin h.T) (k : Fin h.width) :
    ∃ p, p∈phases h ∧ ∃ prog : Program h p, ∃ step : Fin p.span, ∃ lane : Fin h.BK,
      prog.row=row.val ∧ time p prog.chunk step.val=t.val ∧ column h prog.feature lane.val=k.val := by
  obtain ⟨p,hp,chunk,step,ht⟩ := time_owner h t
  have feature : k.val/h.BK<h.features :=
    (Nat.div_lt_iff_lt_mul h.featureTilePositive).mpr k.isLt
  refine ⟨p,hp,⟨k.val/h.BK,chunk.val,row.val,feature,chunk.isLt,row.isLt⟩,
    step,⟨k.val%h.BK,Nat.mod_lt _ h.featureTilePositive⟩,rfl,ht,?_⟩
  dsimp [column]
  have division := Nat.mod_add_div k.val h.BK
  nlinarith

def programs (h : Call) (p : Phase) : List (Program h p) :=
  (List.finRange h.features).flatMap fun feature =>
    (List.finRange p.count).flatMap fun chunk =>
      (List.finRange h.rows).map fun row =>
        ⟨feature.val,chunk.val,row.val,feature.isLt,chunk.isLt,row.isLt⟩

theorem all_programs (h : Call) (p : Phase) (prog : Program h p) : prog∈programs h p := by
  apply List.mem_flatMap.mpr
  refine ⟨⟨prog.feature,prog.featureInside⟩,List.mem_finRange _,?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨⟨prog.chunk,prog.chunkInside⟩,List.mem_finRange _,?_⟩
  apply List.mem_map.mpr
  refine ⟨⟨prog.row,prog.rowInside⟩,List.mem_finRange _,?_⟩
  cases prog
  rfl

theorem flat_injective (h : Call) (row row' time time' channel channel' : Nat)
    (timeInside : time<h.T) (timeInside' : time'<h.T)
    (channelInside : channel<h.width) (channelInside' : channel'<h.width)
    (eq : row*h.headStride+time*h.width+channel=row'*h.headStride+time'*h.width+channel') :
    row=row' ∧ time=time' ∧ channel=channel' := by
  have width : 0<h.width := Nat.zero_lt_of_lt channelInside
  have localBound : time*h.width+channel<h.headStride := by
    dsimp [Call.headStride]
    nlinarith
  have localBound' : time'*h.width+channel'<h.headStride := by
    dsimp [Call.headStride]
    nlinarith
  have rowEq : row=row' := by
    rcases lt_trichotomy row row' with hr|hr|hr
    · have order := Nat.mul_le_mul_right h.headStride (Nat.succ_le_of_lt hr)
      nlinarith
    · exact hr
    · have order := Nat.mul_le_mul_right h.headStride (Nat.succ_le_of_lt hr)
      nlinarith
  subst row'
  have timeEq : time=time' := by
    rcases lt_trichotomy time time' with ht|ht|ht
    · have order := Nat.mul_le_mul_right h.width (Nat.succ_le_of_lt ht)
      nlinarith
    · exact ht
    · have order := Nat.mul_le_mul_right h.width (Nat.succ_le_of_lt ht)
      nlinarith
  subst time'
  exact ⟨rfl,rfl,by omega⟩

theorem phase_cases (h : Call) (p : Phase) (hp : p∈phases h) :
    p=mainPhase h ∨ p=tailPhase h := by
  simp only [phases,List.mem_append] at hp
  rcases hp with hp|hp
  · left
    split at hp <;> simpa using hp
  · right
    split at hp <;> simpa using hp

theorem main_before_tail (h : Call) (prog : Program h (mainPhase h)) (step : Fin (mainPhase h).span) :
    time (mainPhase h) prog.chunk step.val<(tailPhase h).start := by
  have hc := Nat.mul_le_mul_right h.BT (Nat.succ_le_of_lt prog.chunkInside)
  have hs := step.isLt
  dsimp [time,mainPhase,tailPhase] at *
  nlinarith

theorem same_phase (h : Call) (p p' : Phase) (hp : p∈phases h) (hp' : p'∈phases h)
    (a : Program h p) (b : Program h p') (t : Fin p.span) (t' : Fin p'.span)
    (same : time p a.chunk t.val=time p' b.chunk t'.val) : p=p' := by
  rcases phase_cases h p hp with rfl|rfl <;> rcases phase_cases h p' hp' with rfl|rfl
  · rfl
  · have before := main_before_tail h a t
    dsimp [time] at same before
    omega
  · have before := main_before_tail h b t'
    dsimp [time] at same before
    omega
  · rfl

theorem chunk_unique (p : Phase) (a b : Nat) (t t' : Fin p.span)
    (same : time p a t.val=time p b t'.val) : a=b ∧ t.val=t'.val := by
  have ht := t.isLt
  have ht' := t'.isLt
  have positive : 0<p.span := Nat.zero_lt_of_lt ht
  have chunks : a=b := by
    rcases lt_trichotomy a b with lt|eq|lt
    · have order := Nat.mul_le_mul_right p.span (Nat.succ_le_of_lt lt)
      dsimp [time] at same
      nlinarith
    · exact eq
    · have order := Nat.mul_le_mul_right p.span (Nat.succ_le_of_lt lt)
      dsimp [time] at same
      nlinarith
  subst b
  exact ⟨rfl,by dsimp [time] at same; omega⟩

theorem feature_unique (h : Call) (a b : Nat) (j j' : Fin h.BK)
    (same : column h a j.val=column h b j'.val) : a=b ∧ j.val=j'.val := by
  have ht := j.isLt
  have ht' := j'.isLt
  have chunks : a=b := by
    rcases lt_trichotomy a b with lt|eq|lt
    · have order := Nat.mul_le_mul_right h.BK (Nat.succ_le_of_lt lt)
      dsimp [column] at same
      nlinarith
    · exact eq
    · have order := Nat.mul_le_mul_right h.BK (Nat.succ_le_of_lt lt)
      dsimp [column] at same
      nlinarith
  subst b
  exact ⟨rfl,by dsimp [column] at same; omega⟩

theorem offset_unique (h : Call) (p p' : Phase) (hp : p∈phases h) (hp' : p'∈phases h)
    (a : Program h p) (b : Program h p') (t : Fin p.span) (t' : Fin p'.span)
    (j j' : Fin h.BK) (same : offset h p a t.val j.val=offset h p' b t'.val j'.val) :
    p=p' ∧ a.row=b.row ∧ a.feature=b.feature ∧ a.chunk=b.chunk ∧ t.val=t'.val ∧ j.val=j'.val := by
  obtain ⟨rows,times,channels⟩ := flat_injective h a.row b.row _ _ _ _
    (time_inside h p hp a t) (time_inside h p' hp' b t')
    (column_inside h p a j) (column_inside h p' b j') same
  have phaseEq := same_phase h p p' hp hp' a b t t' times
  subst p'
  obtain ⟨chunks,steps⟩ := chunk_unique p a.chunk b.chunk t t' times
  obtain ⟨features,lanes⟩ := feature_unique h a.feature b.feature j j' channels
  exact ⟨rfl,rows,features,chunks,steps,lanes⟩

#axiomsClean offset_unique
#axiomsClean coordinate_owner
#axiomsClean all_programs
#axiomsClean flat_injective
end VeriTile.Bench.Repairs.DecayCumsum.Coverage
