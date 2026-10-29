import bench.repairs.tritonbench_g.quantize_copy_kv.IO
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 4500000

theorem initial_eq {A : Type} (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (version : Bool) (token : Nat) (s : State A) :
    (invocation M convert equal p version token).initial s = KernelGrid.atProgram s token := by
  simp [invocation, Invocation.initial, bindRegisters, KernelGrid.atProgram]
  funext ax
  exact (Function.update_apply s.pids 0 token ax).symm

theorem run_eq {A : Type} [Inhabited A] (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (version : Bool) (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (invocation M convert equal p version)) s =
      KernelGrid.run (KernelExecution.backend M convert equal) p version jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j jobs ih =>
    simp only [List.map_cons, runSchedule, KernelGrid.run, Invocation.run, initial_eq]
    change Option.bind _ _ = Option.bind _ _
    congr 1
    funext t
    exact ih t

theorem metadata_binding {A : Type} [Inhabited A]
    (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool)
    (values : Inputs A p count) (domain : metadataMatches p count dest values) (s : State A)
    (binding : (programIO M convert equal p count dest version).InputBinding values s)
    (token : Nat) (active : token < count) :
    (s.mem "destination" token).read .int = (dest token : Int) :=
  (binding ⟨0, by simp [programIO, memory, inputs]⟩ (⟨token, active⟩, PUnit.unit) trivial).trans
    (domain ⟨token, active⟩)

theorem payload_binding {A : Type} [Inhabited A]
    (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool)
    (values : Inputs A p count) (s : State A)
    (binding : (programIO M convert equal p count dest version).InputBinding values s)
    (token head column : Nat) (ht : token < count) (hh : head < p.heads) (hc : column < p.width) :
    (s.mem "input" (KernelResult.inputOffset p token head column)).read .real = payload values token head column := by
  simpa [payload, ht, hh, hc] using binding ⟨1, by simp [programIO, memory, inputs]⟩
    (⟨token, ht⟩, ⟨head, hh⟩, ⟨column, hc⟩, PUnit.unit) trivial

theorem schedule_safe {A : Type} [Inhabited A]
    (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool) (width : 0 < p.width)
    (values : Inputs A p count) (domain : metadataMatches p count dest values)
    (bounds : RegionName → Nat) (hb : (programIO M convert equal p count dest version).InBounds bounds values)
    (initial : State A) (binding : (programIO M convert equal p count dest version).InputBinding values initial)
    (jobs : List Nat) (valid : ∀ token ∈ jobs, token < count)
    (s : State A) (same : KernelGridResult.SameInputs initial s) :
    ScheduleSafe bounds (jobs.map (invocation M convert equal p version)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons token jobs ih =>
    have ht := valid token (by simp)
    have oldMeta := metadata_binding M convert equal p count dest version values domain initial binding token ht
    have metadataCurrent := KernelGridResult.destination_read initial s same token (dest token) oldMeta
    refine ⟨?_, ?_⟩
    · change FP.Structural.TraceSafe _ bounds _ ((invocation M convert equal p version token).initial s)
      rw [initial_eq]
      apply KernelExecution.source_traceSafe M convert equal p version _ bounds (dest token) width
      · simpa [KernelGrid.atProgram] using metadataCurrent
      · simpa [KernelGrid.atProgram] using hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩
          (⟨token, ht⟩, PUnit.unit) trivial
      · intro h hh c
        simpa [KernelGrid.atProgram] using hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩
          (⟨token, ht⟩, ⟨h.val, hh⟩, c, PUnit.unit) trivial
      · intro h hh c
        exact hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩
          (⟨token, ht⟩, ⟨h.val, hh⟩, c, PUnit.unit) trivial
      · intro h hh
        exact hb.2.2.1 ⟨1, by simp [programIO, memory, outputs]⟩
          (⟨token, ht⟩, ⟨h.val, hh⟩, PUnit.unit) trivial
    · intro t ran
      have expected : (invocation M convert equal p version token).run s =
          some (KernelGridResult.advance M convert equal p width s token) := by
        unfold Invocation.run
        rw [initial_eq]
        cases version
        · exact (KernelExecution.execution M convert equal p (KernelGrid.atProgram s token) width).property
        · exact (KernelExecution.same_run (KernelExecution.backend M convert equal) p
            (KernelGrid.atProgram s token)).symm.trans
            (KernelExecution.execution M convert equal p (KernelGrid.atProgram s token) width).property
      have sameState := Option.some.inj (expected.symm.trans ran)
      subst t
      exact ih (fun t h => valid t (by simp [h])) _
        (KernelGridResult.advance_inputs M convert equal p width initial s same token (dest token) oldMeta)

theorem implements {A : Type} [Inhabited A]
    (M : Algebra A) (convert : A → Int) (equal : A → A → Bool)
    (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool) (width : 0 < p.width)
    (codes : Inputs A p count → Nat → Nat → Nat → Int)
    (scales : Inputs A p count → Nat → Nat → A)
    (correct : ∀ (values : Inputs A p count) (s : State A),
      metadataMatches p count dest values →
      (programIO M convert equal p count dest version).InputBinding values s →
      ∃ final,
        KernelGrid.run (KernelExecution.backend M convert equal) p Bool.false (List.range count) s = some final ∧
        KernelGrid.run (KernelExecution.backend M convert equal) p Bool.true (List.range count) s = some final ∧
        RawKernel.Result p count dest (codes values) (scales values) s final) :
    (programIO M convert equal p count dest version).ImplementsOn (metadataMatches p count dest)
      (fun values => result (codes values) (scales values)) := by
  intro values domain bounds hb s binding
  refine ⟨schedule_safe M convert equal p count dest version width values domain bounds hb s binding
    (List.range count) (fun t h => List.mem_range.mp h) s ⟨fun _ => rfl, fun _ => rfl⟩, ?_⟩
  obtain ⟨final, leftRun, rightRun, result⟩ := correct values s domain binding
  refine ⟨final, ?_, ?_, ?_⟩
  · rw [show (programIO M convert equal p count dest version).schedule values =
      (List.range count).map (invocation M convert equal p version) from rfl, run_eq]
    cases version
    · exact leftRun
    · exact rightRun
  · intro o j _
    fin_cases o
    · simpa [mapCell, mapValue] using congrArg (mapCell (id : A → A))
        (result.1 j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt)
    · simpa [mapCell, mapValue] using congrArg (mapCell (id : A → A))
        (result.2.1 j.1.val j.2.1.val j.1.isLt j.2.1.isLt)
  · intro r offset outside
    apply result.2.2
    · by_cases hr : r = "output"
      · right
        intro token head column ht hh hc
        exact (outside.1 ⟨0, by simp [programIO, memory, outputs]⟩
          (⟨token, ht⟩, ⟨head, hh⟩, ⟨column, hc⟩, PUnit.unit) trivial).resolve_left
          (by simpa [outputs] using not_ne_iff.mpr hr)
      · exact Or.inl hr
    · by_cases hr : r = "scale"
      · right
        intro token head ht hh
        exact (outside.1 ⟨1, by simp [programIO, memory, outputs]⟩
          (⟨token, ht⟩, ⟨head, hh⟩, PUnit.unit) trivial).resolve_left
          (by simpa [outputs] using not_ne_iff.mpr hr)
      · exact Or.inl hr

#axiomsClean initial_eq
#axiomsClean run_eq
#axiomsClean schedule_safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Interface
