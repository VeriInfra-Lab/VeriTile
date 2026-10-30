import bench.tritonbench_g.flash_decode2_phi.IO
import bench.tritonbench_g.flash_decode2_llama.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.PhiIO
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000

/-- Every original finite-grid length assignment has a backing extent. This
does not impose Q>0, positive lengths, or an artificial count≤source-S guard. -/
theorem exists_capacity (B H Q : Nat) (length : Nat→Nat) :
    ∃ cap,∀ row,row<B*H → iterations Q (length (row/H))≤cap := by
  refine ⟨(Finset.range (B*H)).sup (fun row => iterations Q (length (row/H))),?_⟩
  intro row hr
  exact Finset.le_sup (f:=fun row : Nat => iterations Q (length (row/H))) (Finset.mem_range.mpr hr)

theorem schedule_run {β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (s : State β) :
    runSchedule (schedule A g order) s=
      Grid.source A Bool.true g.B g.H g.S g.D g.Q g.BD order s := by
  induction order generalizing s with
  | nil => rfl
  | cons row rest ih =>
    change (FP.Structural.exec A (FlashDecode2.kernel Bool.true g.H g.S g.D g.Q g.BD)
      (Grid.setup g.B g.H row s)).bind _ = _
    rw [Grid.source]
    congr 1
    funext next
    exact ih next

theorem batch_bound (g : Geometry) (row : Nat) (hr : row<g.B*g.H) : row/g.H<g.B := by
  have positive : 0<g.H := by nlinarith
  exact (Nat.div_lt_iff_lt_mul positive).mpr hr

theorem length_binding {α β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (encode : α→β) (observe : β→α)
    (xs : Inputs α g) (s : State β) (pins : (decodeIO A g order encode observe).InputBinding xs s)
    (b : Nat) (hb : b<g.B) : (s.mem "L" b).read .nat=lengths xs b := by
  simpa [lengths,hb,mapValue] using pins (2 : Fin 3) (⟨b,hb⟩,PUnit.unit) trivial

theorem count_binding {α β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (encode : α→β) (observe : β→α)
    (xs : Inputs α g) (s : State β) (pins : (decodeIO A g order encode observe).InputBinding xs s)
    (row : Nat) (hr : row<g.B*g.H) : count (Grid.setup g.B g.H row s) g.Q=steps xs row := by
  have length := length_binding A g order encode observe xs s pins _ (batch_bound g row hr)
  simp [count,Grid.setup,steps,iterations,length]

theorem log_binding {α β : Type} [Inhabited α] [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (encode : α→β) (observe : β→α)
    (xs : Inputs α g) (s : State β) (pins : (decodeIO A g order encode observe).InputBinding xs s)
    (row j : Nat) (hr : row<g.B*g.H) (hj : j<g.capacity) (used : j<steps xs row) :
    (s.mem "E" (row*g.S+j)).read .real=encode (logs xs row j) := by
  simpa [logs,hr,hj,mapValue] using
    pins (0 : Fin 3) (⟨row,hr⟩,⟨j,hj⟩,PUnit.unit) used

theorem mid_binding {α β : Type} [Inhabited α] [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (encode : α→β) (observe : β→α)
    (xs : Inputs α g) (s : State β) (pins : (decodeIO A g order encode observe).InputBinding xs s)
    (row j d : Nat) (hr : row<g.B*g.H) (hj : j<g.capacity) (used : j<steps xs row) (hd : d<g.D) :
    (s.mem "M" (row*g.S*g.D+j*g.D+d)).read .real=encode (mids xs row j d) := by
  simpa [mids,hr,hj,hd,mapValue] using
    pins (1 : Fin 3) (⟨row,hr⟩,⟨j,hj⟩,⟨d,hd⟩,PUnit.unit) used

theorem schedule_safe {α β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (xs : Inputs α g) (order : List Nat)
    (inside : ∀ row∈order,row<g.B*g.H) (bounds : RegionName→Nat)
    (length : ∀ b,b<g.B → b<bounds "L")
    (log : ∀ row j,row<g.B*g.H → j<steps xs row → row*g.S+j<bounds "E")
    (mid : ∀ row j d,row<g.B*g.H → j<steps xs row → d<g.D → row*g.S*g.D+j*g.D+d<bounds "M")
    (out : ∀ row d,row<g.B*g.H → d<g.D → row*g.D+d<bounds "O")
    (s : State β) (counts : ∀ row,row<g.B*g.H → count (Grid.setup g.B g.H row s) g.Q=steps xs row) :
    ScheduleSafe bounds (schedule A g order) s := by
  induction order generalizing s with
  | nil => trivial
  | cons row rest ih =>
    have hr := inside row (by simp)
    constructor
    · change FP.Structural.TraceSafe A bounds (FlashDecode2.kernel Bool.true g.H g.S g.D g.Q g.BD)
        (Grid.setup g.B g.H row s)
      apply FlashDecode2.AccessSafety.source_safe
      · exact length _ (batch_bound g row hr)
      · intro j hj k active
        rw [row_mid_offset]
        exact mid row j (lane k) hr ((counts row hr) ▸ hj) (by simpa [FlashDecode2.active] using active)
      · intro j hj
        rw [row_log_offset]
        exact log row j hr ((counts row hr) ▸ hj)
      · intro k active
        rw [Grid.output_offset]
        exact out row (lane k) hr (by simpa [FlashDecode2.active] using active)
    · intro next executed
      obtain ⟨actual,ran,memory⟩ := complete A (Grid.setup g.B g.H row s)
        Bool.true g.H g.S g.D g.Q g.BD
      change FP.Structural.exec A (FlashDecode2.kernel Bool.true g.H g.S g.D g.Q g.BD)
        (Grid.setup g.B g.H row s)=some next at executed
      rw [ran] at executed
      cases executed
      apply ih (fun r h => inside r (by simp [h])) _
      intro r hin
      have same (i : Nat) : next.mem "L" i=s.mem "L" i := by
        rw [memory]
        exact scatter_frame _ g.H g.D g.BD Bool.true _ "L" i (Or.inl (by decide))
      simpa only [count,Grid.setup,same] using counts r hin

theorem safe {α β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (order : List Nat) (coverage : order.Perm (List.range (g.B*g.H)))
    (encode : α→β) (observe : β→α) (xs : Inputs α g) (fits : Fits g xs)
    (bounds : RegionName→Nat) (covered : (decodeIO A g order encode observe).InBounds bounds xs)
    (s : State β) (pins : (decodeIO A g order encode observe).InputBinding xs s) :
    ScheduleSafe bounds (schedule A g order) s := by
  apply schedule_safe A g xs order (fun row h => List.mem_range.mp (coverage.mem_iff.mp h)) bounds
  · intro b hb
    exact covered.2.1 (2 : Fin 3) (⟨b,hb⟩,PUnit.unit) trivial
  · intro row j hr hj
    exact covered.2.1 (0 : Fin 3) (⟨row,hr⟩,⟨j,lt_of_lt_of_le hj (fits row hr)⟩,PUnit.unit) hj
  · intro row j d hr hj hd
    exact covered.2.1 (1 : Fin 3)
      (⟨row,hr⟩,⟨j,lt_of_lt_of_le hj (fits row hr)⟩,⟨d,hd⟩,PUnit.unit) hj
  · intro row d hr hd
    exact covered.2.2.1 (0 : Fin 1) (⟨row,hr⟩,⟨d,hd⟩,PUnit.unit) trivial
  · exact count_binding A g order encode observe xs s pins

theorem outside {α β : Type} (A : Algebra β) (g : Geometry) (order : List Nat)
    (encode : α→β) (observe : β→α) (xs : Inputs α g) (r : RegionName) (o : Nat)
    (h : (decodeIO A g order encode observe).Outside xs r o) : r≠"O" ∨ g.B*g.H*g.D≤o := by
  by_cases region : r="O"
  · right
    by_contra notBound
    have below : o<g.B*g.H*g.D := by omega
    have hp : 0<g.D := by
      by_contra hz
      have zero : g.D=0 := by omega
      simp [zero] at below
    have hr : o/g.D<g.B*g.H := (Nat.div_lt_iff_lt_mul hp).mpr below
    have hd := Nat.mod_lt o hp
    have ne := (h.1 (0 : Fin 1) (⟨o/g.D,hr⟩,⟨o%g.D,hd⟩,PUnit.unit) trivial).resolve_left (not_not.mpr region)
    apply ne
    change o=o/g.D*g.D+o%g.D
    simpa only [Nat.mul_comm] using (Nat.div_add_mod o g.D).symm
  · exact Or.inl region

/-- Reuse the complete source run, full typed outputs and exact raw frame.
The result lemma supplies only an independent logical value for each output. -/
theorem implements {α β : Type} [Inhabited β] (A : Algebra β)
    (g : Geometry) (width : g.D≤g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) (encode : α→β) (observe : β→α)
    (domain : Inputs α g→Prop) (fits : ∀ xs,domain xs→Fits g xs)
    (f : Inputs α g→Fin (g.B*g.H)→Fin g.D→α)
    (value : ∀ xs,domain xs→∀ s,(decodeIO A g order encode observe).InputBinding xs s→
      ∀ (row : Fin (g.B*g.H)) (d : Fin g.D),
      observe (Grid.value A s Bool.true g.B g.H g.S g.D g.Q g.BD row.val
        (⟨d.val,lt_of_lt_of_le d.isLt width⟩,PUnit.unit))=f xs row d) :
    (decodeIO A g order encode observe).ImplementsOn domain (fun xs => result (f xs)) := by
  intro xs hd bounds hb initial pins
  refine ⟨safe A g order coverage encode observe xs (fits xs hd) bounds hb initial pins,?_⟩
  obtain ⟨final,ran,outputs,frame⟩ := Raw.complete A g.B g.H g.S g.D g.Q g.BD order coverage initial
  refine ⟨final,(schedule_run A g order initial).trans ran,?_,?_⟩
  · intro o
    refine Fin.cases ?_ (fun i => i.elim0) o
    intro j _
    change mapCell observe (final.mem "O" (j.1.val*g.D+j.2.1.val))=_
    rw [outputs j.1.val (⟨j.2.1.val,lt_of_lt_of_le j.2.1.isLt width⟩,PUnit.unit) j.1.isLt j.2.1.isLt]
    change Cell.mk .real (observe _)=Cell.mk .real _
    rw [value xs hd initial pins j.1 j.2.1]
    rfl
  · intro r o h
    exact frame r o (outside A g order encode observe xs r o h)

#axiomsClean exists_capacity
#axiomsClean implements
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.PhiIO
