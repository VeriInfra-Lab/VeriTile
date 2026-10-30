import bench.repairs.tritonbench_g.quantize_kv_copy.IO
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.PublicIO
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
set_option autoImplicit true
variable [Inhabited α]

theorem initial_eq (op : Ops α) (p : Parameters) (pid : Nat) (s : State α) :
    (invocation op p pid).initial s = KernelGrid.atProgram p s pid := by
  simp only [Invocation.initial, invocation, bindRegisters, List.foldl_nil,
    Bool.false_eq_true, if_false, Option.getD_none, KernelGrid.atProgram]
  congr 1
  funext d
  by_cases h0 : d=0
  · subst d; rfl
  by_cases h1 : d=1
  · subst d; rfl
  have hd : ¬d<2 := by omega
  simp [h0,h1,hd]

theorem invocation_run (op : Ops α) (p : Parameters) (pid : Nat) (s : State α) :
    (invocation op p pid).run s =
      FP.Structural.exec (backend op) (kernel p) (KernelGrid.atProgram p s pid) := by
  unfold Invocation.run
  rw [initial_eq]
  rfl

theorem schedule_eq (op : Ops α) (p : Parameters) (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (invocation op p)) s = KernelGrid.run op p jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,runSchedule,invocation_run,KernelGrid.run]
    cases h : FP.Structural.exec (backend op) (kernel p) (KernelGrid.atProgram p s pid) <;> simp [ih]

theorem input_reads (op : Ops α) (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (values : Inputs α p n) (hm : Metadata dest values) (s : State α)
    (hin : (sourceIO op p n dest).InputBinding values s) :
    KernelRaw.Reads p n dest (elements values) s := by
  constructor
  · intro pid g c hp hg hc
    have h := hin ⟨0,by simp [sourceIO,inputs]⟩
      (⟨pid,hp⟩,⟨g,hg⟩,⟨c,hc⟩,PUnit.unit) trivial
    simpa only [elements,dif_pos hp,dif_pos hg,dif_pos hc] using h
  · intro pid hp
    exact (hin ⟨1,by simp [sourceIO,inputs]⟩ (⟨pid,hp⟩,PUnit.unit) trivial).trans (hm ⟨pid,hp⟩)

theorem schedule_safe (op : Ops α) (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (values : Inputs α p n) (bounds : RegionName → Nat)
    (hb : (sourceIO op p n dest).InBounds bounds values)
    (initial : State α) (reads : KernelRaw.Reads p n dest (elements values) initial)
    (jobs : List Nat) (valid : ∀ pid∈jobs,pid<n)
    (s : State α) (same : KernelGridResult.SameInputs initial s) :
    ScheduleSafe bounds (jobs.map (invocation op p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    have hp := valid pid (by simp)
    have hd := KernelGridResult.destination_read p initial s same pid (dest pid) (reads.destination pid hp)
    have pos : pid/p.heads*p.heads+pid%p.heads=pid := by
      simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div pid p.heads
    constructor
    · change FP.Structural.TraceSafe (backend op) bounds (kernel p) ((invocation op p pid).initial s)
      rw [initial_eq]
      apply AccessSafety.source_safe _ _ _ _ (dest pid)
      · simpa [KernelGrid.atProgram] using hd
      · exact hb.2.1 ⟨1,by simp [sourceIO,inputs]⟩ (⟨pid,hp⟩,PUnit.unit) trivial
      · intro g c hg hc
        simpa [KernelGrid.atProgram,pos] using hb.2.1 ⟨0,by simp [sourceIO,inputs]⟩
          (⟨pid,hp⟩,⟨g,hg⟩,⟨c,hc⟩,PUnit.unit) trivial
      · intro g c hg hc
        exact hb.2.2.1 ⟨0,by simp [sourceIO,outputs]⟩
          (⟨pid,hp⟩,⟨g,hg⟩,⟨c,hc⟩,PUnit.unit) trivial
      · intro g hg
        exact hb.2.2.1 ⟨1,by simp [sourceIO,outputs]⟩ (⟨pid,hp⟩,⟨g,hg⟩,PUnit.unit) trivial
    · intro next run
      rw [invocation_run] at run
      have run' := (execution op p (KernelGrid.atProgram p s pid)).property
      have eq : next = KernelGridResult.advance op p s pid := Option.some.inj (run.symm.trans run')
      subst next
      exact ih (fun q hq => valid q (by simp [hq])) _
        (KernelGridResult.advance_inputs op p initial s same pid (dest pid) (reads.destination pid hp))

/-- Assemble the reused whole source result with actual trace safety and the
shared typed output/frame judgment. The source runner is fixed by the IO. -/
theorem implements (op : Ops α) (p : Parameters) (n : Nat) (dest : Nat → Nat)
    (codes : Inputs α p n → Nat → Nat → Nat → Int)
    (scales : Inputs α p n → Nat → Nat → α)
    (correct : ∀ values s, KernelRaw.Reads p n dest (elements values) s →
      ∃ final, KernelGrid.run op p (List.range n) s=some final ∧
        KernelRaw.Result p n dest (codes values) (scales values) s final) :
    (sourceIO op p n dest).ImplementsOn (Metadata dest)
      (fun values => result (codes values) (scales values)) := by
  intro values hm bounds hb initial hin
  have reads := input_reads op p n dest values hm initial hin
  refine ⟨schedule_safe op p n dest values bounds hb initial reads (List.range n)
    (fun _ h => List.mem_range.mp h) initial ⟨fun _ => rfl,fun _ => rfl⟩,?_⟩
  obtain ⟨final,ran,codesOut,scalesOut,frame⟩ := correct values initial reads
  refine ⟨final,(schedule_eq op p (List.range n) initial).trans ran,?_,?_⟩
  · intro o j _
    fin_cases o
    · change mapCell id (final.mem "O" (codeOffset p (KernelGridResult.target p dest j.1.val) j.2.1.val j.2.2.1.val)) =
        Cell.mk .int (codes values j.1.val j.2.1.val j.2.2.1.val)
      rw [codesOut j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt]
      rfl
    · change mapCell id (final.mem "S" (scaleOffset p (KernelGridResult.target p dest j.1.val) j.2.1.val)) =
        Cell.mk .real (scales values j.1.val j.2.1.val)
      rw [scalesOut j.1.val j.2.1.val j.1.isLt j.2.1.isLt]
      rfl
  · intro r address outside
    apply frame r address
    · by_cases hr : r="O"
      · right
        intro pid g c hp hg hc
        exact (outside.1 ⟨0,by simp [sourceIO,outputs]⟩
          (⟨pid,hp⟩,⟨g,hg⟩,⟨c,hc⟩,PUnit.unit) trivial).resolve_left (by simpa using hr)
      · exact Or.inl hr
    · by_cases hr : r="S"
      · right
        intro pid g hp hg
        exact (outside.1 ⟨1,by simp [sourceIO,outputs]⟩
          (⟨pid,hp⟩,⟨g,hg⟩,PUnit.unit) trivial).resolve_left (by simpa using hr)
      · exact Or.inl hr

#axiomsClean schedule_eq
#axiomsClean schedule_safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.PublicIO
