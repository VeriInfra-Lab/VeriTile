import bench.tritonbench_g.flash_decode2_llama.IO
import bench.tritonbench_g.flash_decode2_llama.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.PublicIO
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

theorem launch_initial {β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (row : Nat) (s : State β) :
    (IO.launch A g row).initial s = Grid.setup g.B g.H row s := rfl

theorem schedule_run {β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat) (s : State β) :
    runSchedule (IO.schedule A g order) s =
      Grid.source A Bool.false g.B g.H g.S g.D g.Q g.D order s := by
  induction order generalizing s with
  | nil => rfl
  | cons row rest ih =>
    change ((IO.launch A g row).run s).bind _ = _
    change (FP.Structural.exec A (FlashDecode2.kernel Bool.false g.H g.S g.D g.Q g.D)
      (Grid.setup g.B g.H row s)).bind _ = _
    rw [Grid.source]
    congr 1
    funext next
    exact ih next

theorem batch_bound (g : KernelInterface.Geometry) (row : Nat) (hr : row < g.B * g.H) :
    row / g.H < g.B := by
  have hp : 0 < g.H := by nlinarith
  exact (Nat.div_lt_iff_lt_mul hp).mpr hr

theorem length_binding {α β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (s : State β)
    (pins : (IO.decodeIO A g order encode observe).InputBinding vals s)
    (batch : Nat) (hb : batch < g.B) :
    (s.mem "L" batch).read .nat = IO.lengths vals batch := by
  simpa [IO.lengths, hb, mapValue] using pins (2 : Fin 3) (⟨batch, hb⟩, PUnit.unit) trivial

theorem lengths_bound {α β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (domain : IO.Lengths g vals) (s : State β)
    (pins : (IO.decodeIO A g order encode observe).InputBinding vals s) : LengthBounds g s := by
  intro row hr
  rw [length_binding A g order encode observe vals s pins _ (batch_bound g row hr)]
  exact domain row hr

theorem log_binding {α β : Type} [Inhabited α] [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (s : State β)
    (pins : (IO.decodeIO A g order encode observe).InputBinding vals s)
    (row j : Nat) (hr : row < g.B * g.H) (hj : j < g.S) :
    (s.mem "E" (row * g.S + j)).read .real = encode (IO.logValues vals row j) := by
  simpa [IO.logValues, hr, hj, mapValue] using
    pins (0 : Fin 3) (⟨row, hr⟩, ⟨j, hj⟩, PUnit.unit) trivial

theorem mid_binding {α β : Type} [Inhabited α] [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (s : State β)
    (pins : (IO.decodeIO A g order encode observe).InputBinding vals s)
    (row j d : Nat) (hr : row < g.B * g.H) (hj : j < g.S) (hd : d < g.D) :
    (s.mem "M" (row * g.S * g.D + j * g.D + d)).read .real =
      encode (IO.midValues vals row j d) := by
  simpa [IO.midValues, hr, hj, hd, mapValue] using
    pins (1 : Fin 3) (⟨row, hr⟩, ⟨j, hj⟩, ⟨d, hd⟩, PUnit.unit) trivial

theorem schedule_safe {β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat)
    (inside : ∀ row ∈ order, row < g.B * g.H) (bounds : RegionName → Nat)
    (length : ∀ batch, batch < g.B → batch < bounds "L")
    (log : ∀ row j, row < g.B * g.H → j < g.S → row * g.S + j < bounds "E")
    (mid : ∀ row j d, row < g.B * g.H → j < g.S → d < g.D →
      row * g.S * g.D + j * g.D + d < bounds "M")
    (out : ∀ row d, row < g.B * g.H → d < g.D → row * g.D + d < bounds "O")
    (s : State β) (domain : LengthBounds g s) : ScheduleSafe bounds (IO.schedule A g order) s := by
  induction order generalizing s with
  | nil => trivial
  | cons row rest ih =>
    have hr := inside row (by simp)
    have counted := KernelInterface.count_bound g s domain row hr
    constructor
    · change FP.Structural.TraceSafe A bounds (FlashDecode2.kernel Bool.false g.H g.S g.D g.Q g.D)
        (Grid.setup g.B g.H row s)
      apply FlashDecode2.AccessSafety.source_safe
      · exact length _ (batch_bound g row hr)
      · intro j hj k _
        rw [row_mid_offset]
        exact mid row j (lane k) hr (lt_of_lt_of_le (counted.1 ▸ hj) counted.2.2) k.1.isLt
      · intro j hj
        rw [row_log_offset]
        exact log row j hr (lt_of_lt_of_le (counted.1 ▸ hj) counted.2.2)
      · intro k _
        rw [Grid.output_offset]
        exact out row (lane k) hr k.1.isLt
    · intro next executed
      obtain ⟨actual, ran, memory⟩ := complete A (Grid.setup g.B g.H row s)
        Bool.false g.H g.S g.D g.Q g.D
      change FP.Structural.exec A (FlashDecode2.kernel Bool.false g.H g.S g.D g.Q g.D)
        (Grid.setup g.B g.H row s) = some next at executed
      rw [ran] at executed
      cases executed
      apply ih (fun r h => inside r (by simp [h])) _
      intro r hin
      have same (i : Nat) : next.mem "L" i = s.mem "L" i := by
        rw [memory]
        exact scatter_frame _ g.H g.D g.D Bool.false _ "L" i (Or.inl (by decide))
      simpa only [same] using domain r hin

theorem safe {α β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (domain : IO.Lengths g vals) (bounds : RegionName → Nat)
    (covered : (IO.decodeIO A g order encode observe).InBounds bounds vals)
    (s : State β) (pins : (IO.decodeIO A g order encode observe).InputBinding vals s) :
    ScheduleSafe bounds (IO.schedule A g order) s := by
  apply schedule_safe A g order
    (fun row member => List.mem_range.mp (coverage.mem_iff.mp member)) bounds
  · intro b hb
    exact covered.2.1 (2 : Fin 3) (⟨b, hb⟩, PUnit.unit) trivial
  · intro row j hr hj
    exact covered.2.1 (0 : Fin 3) (⟨row, hr⟩, ⟨j, hj⟩, PUnit.unit) trivial
  · intro row j d hr hj hd
    exact covered.2.1 (1 : Fin 3) (⟨row, hr⟩, ⟨j, hj⟩, ⟨d, hd⟩, PUnit.unit) trivial
  · intro row d hr hd
    exact covered.2.2.1 (0 : Fin 1) (⟨row, hr⟩, ⟨d, hd⟩, PUnit.unit) trivial
  · exact lengths_bound A g order encode observe vals domain s pins

theorem outside {α β : Type} (A : Algebra β) (g : KernelInterface.Geometry)
    (order : List Nat) (encode : α → β) (observe : β → α)
    (vals : IO.Inputs α g) (r : RegionName) (o : Nat)
    (h : (IO.decodeIO A g order encode observe).Outside vals r o) :
    r ≠ "O" ∨ g.B * g.H * g.D ≤ o := by
  by_cases region : r = "O"
  · right
    by_contra notBound
    have below : o < g.B * g.H * g.D := by omega
    have hp : 0 < g.D := by
      by_contra hzero
      have zero : g.D = 0 := by omega
      simp [zero] at below
    have hr : o / g.D < g.B * g.H := (Nat.div_lt_iff_lt_mul hp).mpr below
    have hd := Nat.mod_lt o hp
    have ne := (h.1 (0 : Fin 1) (⟨o / g.D, hr⟩, ⟨o % g.D, hd⟩, PUnit.unit)
      trivial).resolve_left (not_not.mpr region)
    apply ne
    change o = o / g.D * g.D + o % g.D
    simpa only [Nat.mul_comm] using (Nat.div_add_mod o g.D).symm
  · exact Or.inl region

/-- Assembly uses the fixed source runner, complete source results and safety.
The logical formula is supplied by a proven per-row observation lemma. -/
theorem implements {α β : Type} [Inhabited β] (A : Algebra β)
    (g : KernelInterface.Geometry) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) (encode : α → β) (observe : β → α)
    (f : IO.Inputs α g → Fin (g.B * g.H) → Fin g.D → α)
    (result : ∀ vals, IO.Lengths g vals → ∀ s,
      (IO.decodeIO A g order encode observe).InputBinding vals s →
      ∀ (row : Fin (g.B * g.H)) (dim : Fin g.D),
        observe (Grid.value A s Bool.false g.B g.H g.S g.D g.Q g.D row.val (dim, PUnit.unit)) =
          f vals row dim) :
    (IO.decodeIO A g order encode observe).ImplementsOn (IO.Lengths g)
      (fun vals => IO.outputValues (f vals)) := by
  intro vals domain bounds covered initial pins
  refine ⟨safe A g order coverage encode observe vals domain bounds covered initial pins, ?_⟩
  obtain ⟨final, ran, outputs, frame⟩ := KernelInterface.typed A g order coverage initial
  refine ⟨final, (schedule_run A g order initial).trans ran, ?_, ?_⟩
  · intro o
    refine Fin.cases ?_ (fun i => i.elim0) o
    intro j _
    change mapCell observe (final.mem "O" (j.1.val * g.D + j.2.1.val)) = _
    rw [outputs j.1.val (j.2.1, PUnit.unit) j.1.isLt]
    change Cell.mk .real (observe _) = Cell.mk .real _
    rw [result vals domain initial pins j.1 j.2.1]
    rfl
  · intro r o h
    exact frame r o (outside A g order encode observe vals r o h)

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.PublicIO
