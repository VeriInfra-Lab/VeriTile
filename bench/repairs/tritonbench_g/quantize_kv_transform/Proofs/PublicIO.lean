import bench.repairs.tritonbench_g.quantize_kv_transform.IO
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelExecution
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

theorem launch_initial {α : Type} (A : Algebra α) (p : Parameters) (token : Nat) (s : State α) :
    (IO.launch A p token).initial s = KernelGrid.atProgram s token := by
  simp [IO.launch, Invocation.initial, bindRegisters, KernelGrid.atProgram]
  funext ax
  by_cases h : ax = 0 <;> simp [h, Function.update_apply]

theorem schedule_run {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (order : List Nat) (s : State α) :
    runSchedule (IO.schedule A p order) s = KernelGrid.run A p order s := by
  induction order generalizing s with
  | nil => rfl
  | cons token rest ih =>
    change ((IO.launch A p token).run s).bind _ = _
    simp only [Invocation.run, launch_initial, KernelGrid.run]
    congr 1
    funext next
    exact ih next

theorem input_binding {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (count : Nat) (order : List Nat) (vals : IO.Inputs α p count) (s : State α)
    (pins : (IO.quantizeIO A p count order).InputBinding vals s)
    (t h c : Nat) (ht : t < count) (hh : h < p.heads) (hc : c < p.columns) :
    (s.mem "input" (KernelResult.inputOffset p t h c)).read .real = IO.values vals t h c := by
  simpa [IO.values, ht, hh, hc, mapValue, KernelResult.inputOffset] using
    pins (0 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, ⟨c, hc⟩, PUnit.unit) trivial

theorem destination_binding {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (count : Nat) (order : List Nat) (vals : IO.Inputs α p count)
    (domain : IO.Destinations p count vals) (s : State α)
    (pins : (IO.quantizeIO A p count order).InputBinding vals s)
    (t : Nat) (ht : t < count) :
    (s.mem "destination" t).read .int = (IO.destination vals t : Int) := by
  have value := pins (1 : Fin 2) (⟨t, ht⟩, PUnit.unit) trivial
  change (s.mem "destination" t).read .int = vals 1 (⟨t, ht⟩, PUnit.unit) at value
  rw [value]
  simp [IO.destination, ht, Int.toNat_of_nonneg (domain.1 ⟨t, ht⟩)]

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) (count : Nat)
    (dest : Nat → Nat) (order : List Nat) (inside : ∀ t ∈ order, t < count)
    (bounds : RegionName → Nat)
    (metadataBound : ∀ t, t < count → t < bounds "destination")
    (input : ∀ t h c, t < count → h < p.heads → c < p.columns →
      KernelResult.inputOffset p t h c < bounds "input")
    (codes : ∀ t h c, t < count → h < p.heads → c < p.columns →
      KernelResult.codeOffset p (dest t) h c < bounds "output")
    (scales : ∀ t h, t < count → h < p.heads → KernelResult.scaleOffset p (dest t) h < bounds "scale")
    (s : State α) (metadata : ∀ t, t < count → (s.mem "destination" t).read .int = (dest t : Int)) :
    ScheduleSafe bounds (IO.schedule (backend M convert equal) p order) s := by
  induction order generalizing s with
  | nil => trivial
  | cons token rest ih =>
    have ht := inside token (by simp)
    constructor
    · change FP.Structural.TraceSafe (backend M convert equal) bounds (kernel p)
        ((IO.launch (backend M convert equal) p token).initial s)
      rw [launch_initial]
      apply AccessSafety.source_safe M convert equal p _ hw (dest token)
      · simpa [KernelGrid.atProgram] using metadata token ht
      · exact metadataBound token ht
      · exact fun h c hh hc => input token h c ht hh hc
      · exact fun h c hh hc => codes token h c ht hh hc
      · exact fun h hh => scales token h ht hh
    · intro next ran
      have actual : (IO.launch (backend M convert equal) p token).run s =
          some (KernelGridResult.advance M convert equal p hw s token) := by
        rw [Invocation.run, launch_initial]
        exact (execution M convert equal p (KernelGrid.atProgram s token) hw).property
      rw [actual] at ran
      cases ran
      have same := KernelGridResult.advance_inputs M convert equal p hw s s
        ⟨fun _ => rfl, fun _ => rfl⟩ token (dest token) (metadata token ht)
      apply ih (fun t h => inside t (by simp [h])) _
      intro t ht
      rw [same.2 t]
      exact metadata t ht

theorem safe {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width) (count : Nat)
    (order : List Nat) (coverage : order.Perm (List.range count))
    (vals : IO.Inputs α p count) (domain : IO.Destinations p count vals)
    (bounds : RegionName → Nat)
    (covered : (IO.quantizeIO (backend M convert equal) p count order).InBounds bounds vals)
    (s : State α) (pins : (IO.quantizeIO (backend M convert equal) p count order).InputBinding vals s) :
    ScheduleSafe bounds (IO.schedule (backend M convert equal) p order) s := by
  apply schedule_safe M convert equal p hw count (IO.destination vals) order
    (fun t h => List.mem_range.mp (coverage.mem_iff.mp h)) bounds
  · intro t ht
    exact covered.2.1 (1 : Fin 2) (⟨t, ht⟩, PUnit.unit) trivial
  · intro t h c ht hh hc
    exact covered.2.1 (0 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, ⟨c, hc⟩, PUnit.unit) trivial
  · intro t h c ht hh hc
    exact covered.2.2.1 (0 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, ⟨c, hc⟩, PUnit.unit) trivial
  · intro t h ht hh
    exact covered.2.2.1 (1 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, PUnit.unit) trivial
  · exact destination_binding (backend M convert equal) p count order vals domain s pins

theorem implements {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count))
    (code : (Nat → Nat → Nat → α) → Nat → Fin p.heads → Fin p.columns → Int)
    (scale : (Nat → Nat → Nat → α) → Nat → Fin p.heads → α)
    (result : ∀ (vals : Nat → Nat → Nat → α) (s : State α) (dest : Nat → Nat),
      (∀ t h c, t < count → h < p.heads → c < p.columns →
        (s.mem "input" (KernelResult.inputOffset p t h c)).read .real = vals t h c) →
      (∀ t, t < count → (s.mem "destination" t).read .int = (dest t : Int)) →
      (∀ x y, x < count → y < count → dest x = dest y → x = y) →
      ∃ final, KernelGrid.run (backend M convert equal) p order s = some final ∧
        (∀ t h c, t < count → final.mem "output" (KernelResult.codeOffset p (dest t) h.val c.val) =
          Cell.mk .int (code vals t h c)) ∧
        (∀ t h, t < count → final.mem "scale" (KernelResult.scaleOffset p (dest t) h.val) =
          Cell.mk .real (scale vals t h)) ∧
        (∀ (r : RegionName) o,
          (r ≠ "output" ∨ ∀ t h c, t < count → h < p.heads → c < p.columns →
            o ≠ KernelResult.codeOffset p (dest t) h c) →
          (r ≠ "scale" ∨ ∀ t h, t < count → h < p.heads → o ≠ KernelResult.scaleOffset p (dest t) h) →
          final.mem r o = s.mem r o)) :
    (IO.quantizeIO (backend M convert equal) p count order).ImplementsOn
      (IO.Destinations p count) (fun vals => IO.outputValues
        (fun t h c => code (IO.values vals) t.val h c) (fun t h => scale (IO.values vals) t.val h)) := by
  intro vals domain bounds covered initial pins
  refine ⟨safe M convert equal p hw count order coverage vals domain bounds covered initial pins, ?_⟩
  obtain ⟨final, ran, codes, scales, frame⟩ := result (IO.values vals) initial (IO.destination vals)
    (input_binding _ p count order vals initial pins)
    (destination_binding _ p count order vals domain initial pins) domain.2
  refine ⟨final, (schedule_run _ p order initial).trans ran, ?_, ?_⟩
  · intro output
    refine Fin.cases ?_ (Fin.cases ?_ (fun i => i.elim0)) output
    · intro j _
      change mapCell id (final.mem "output" (KernelResult.codeOffset p
        (IO.destination vals j.1.val) j.2.1.val j.2.2.1.val)) = _
      rw [codes j.1.val j.2.1 j.2.2.1 j.1.isLt]
      rfl
    · intro j _
      change mapCell id (final.mem "scale" (KernelResult.scaleOffset p
        (IO.destination vals j.1.val) j.2.1.val)) = _
      rw [scales j.1.val j.2.1 j.1.isLt]
      rfl
  · intro r o outside
    apply frame r o
    · by_cases same : r = "output"
      · right
        intro t h c ht hh hc
        exact (outside.1 (0 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, ⟨c, hc⟩, PUnit.unit) trivial).resolve_left
          (not_not.mpr same)
      · exact Or.inl same
    · by_cases same : r = "scale"
      · right
        intro t h ht hh
        exact (outside.1 (1 : Fin 2) (⟨t, ht⟩, ⟨h, hh⟩, PUnit.unit) trivial).resolve_left
          (not_not.mpr same)
      · exact Or.inl same

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.PublicIO
