import bench.tritonbench_g.quant_transpose_kernel.IO
import bench.tritonbench_g.quant_transpose_kernel.Proofs.KernelSafety
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

theorem initial (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) (pid : Nat) (s : State α) :
    (IO.launch algebra round p version pid).initial s=KernelGrid.atProgram s pid := by
  simp [IO.launch,IntegerInvocation.initial,Invocation.initial,bindRegisters,KernelGrid.atProgram]
  funext ax
  by_cases h : ax=0 <;> simp [Function.update,h]

theorem schedule_run [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (version : Bool) (order : List Nat) (s : State α) :
    runIntegerSchedule (order.map (IO.launch algebra round p version)) s=
      KernelGrid.run algebra round p version order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
      simp only [List.map_cons,runIntegerSchedule,IntegerInvocation.run,initial]
      change (KernelExecution.exec algebra round p version (KernelGrid.atProgram s pid)).bind _ = _
      rw [KernelGrid.run]
      cases h : KernelExecution.exec algebra round p version (KernelGrid.atProgram s pid) <;>
        simp [ih]

theorem schedule_safe [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (version : Bool) (bounds : RegionName → Nat)
    (vals : (IO.grid algebra round p version).Inputs)
    (bounded : (IO.grid algebra round p version).InBounds bounds vals)
    (order : List Nat) (s : State α) :
    IntegerScheduleSafe bounds (order.map (IO.launch algebra round p version)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid rest ih =>
      refine ⟨?_,fun _ _ => ih _⟩
      simp only [IntegerInvocation.Safe,initial]
      apply KernelSafety.source_safe algebra round p version (KernelGrid.atProgram s pid) bounds
      · intro r c hr hc
        exact bounded.2.1 (0 : Fin 2) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
      · exact bounded.2.1 (1 : Fin 2) PUnit.unit trivial
      · intro r c hr hc
        simpa [IO.grid,Nat.add_comm] using
          bounded.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial

theorem frame (algebra : Algebra α) (round : α → BitVec 64) (p : Parameters)
    (version : Bool) (vals : (IO.grid algebra round p version).Inputs) (before after : State α)
    (preserved : ∀ r o, r ≠ ⟨p.output⟩ ∨ p.M * p.N ≤ o → after.mem r o=before.mem r o) :
    (IO.grid algebra round p version).Frame vals before after := by
  intro r o outside
  apply preserved r o
  by_cases hr : r=⟨p.output⟩
  · right
    by_contra ho
    have bound : o<p.M*p.N := by omega
    obtain ⟨hm,hn,he⟩ := QuantTransposeFinite.decode_bounds p.M p.N o bound
    have nope := outside.1 (0 : Fin 1) (⟨o%p.M,hm⟩,⟨o/p.M,hn⟩,PUnit.unit) trivial
    simpa [IO.grid,hr,he] using nope
  · exact Or.inl hr

/-- Reuse the exact existing full-grid readback, with the public input binding.
No raw-state correctness proposition is hidden in the IO descriptor. -/
theorem implements [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (separated : KernelGrid.Separate p) (version : Bool) :
    (IO.grid algebra round p version).Implements
      (IO.formula algebra round p version (fun x inverse => (KernelResult.quantized algebra round x inverse).toInt)) := by
  apply IntegerProgramIO.Implements.intro
  · intro vals bounds bounded s _
    exact schedule_safe algebra round p version bounds vals bounded _ s
  · intro vals s inputs
    obtain ⟨final,ran,result,preserved⟩ := KernelGrid.correctness algebra round p separated s
    refine ⟨final,?_,?_,frame algebra round p version vals s final preserved⟩
    · rw [show (IO.grid algebra round p version).schedule vals=IO.schedule algebra round p version from rfl]
      unfold IO.schedule
      rw [schedule_run]
      cases version
      · exact ran
      · exact (KernelGrid.same_run algebra round p _ s).symm.trans ran
    · intro o j _
      fin_cases o
      have bound := QuantTransposeFinite.active_encode_bound p.M p.N j.1.val j.2.1.val j.1.isLt j.2.1.isLt
      have decode := QuantTransposeFinite.encode_decode p.M j.1.val j.2.1.val j.1.isLt
      have value := result (j.2.1.val*p.M+j.1.val) bound
      unfold KernelGrid.reference at value
      rw [decode.1,decode.2] at value
      have input := inputs (0 : Fin 2) j trivial
      have inverse := inputs (1 : Fin 2) PUnit.unit trivial
      change (s.mem ⟨p.x⟩ (j.1.val*p.sam+j.2.1.val*p.san)).read .real=vals (0 : Fin 2) j at input
      change (s.mem ⟨p.scale⟩ 0).read .real=vals (1 : Fin 2) PUnit.unit at inverse
      change mapCell id (final.mem ⟨p.output⟩ (j.2.1.val*p.M+j.1.val))=
        Cell.mk .int (KernelResult.quantized algebra round (vals (0 : Fin 2) j)
          (vals (1 : Fin 2) PUnit.unit)).toInt
      rw [value,input,inverse]
      rfl

theorem equivalent [Inhabited α] (algebra : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (separated : KernelGrid.Separate p) :
    (IO.grid algebra round p Bool.false).Equivalent (IO.commuted algebra round p) :=
  IntegerProgramIO.EquivalentOn.of_implementsOn
    (implements algebra round p separated Bool.false)
    (implements algebra round p separated Bool.true)

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.PublicIO
