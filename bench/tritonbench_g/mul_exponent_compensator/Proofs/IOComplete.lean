import bench.tritonbench_g.mul_exponent_compensator.Proofs.SourceSafety
import bench.tritonbench_g.mul_exponent_compensator.Proofs.ScheduledSource

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.IOComplete
open VeriTile.Triton FP.Structural Interface
noncomputable section
variable {α : Type} [Inhabited α]

/-- Composition of actual source safety and complete arbitrary-order execution.
The scalar identification is discharged by the existing arithmetic proofs. -/
theorem implements (M : Algebra α) (f : α → α)
    (identify : ∀ x,scalar M x=f x) (src dst : RegionName) (n B : Nat)
    (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) :
    (io M src dst n B order).Implements (formula f src dst n) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    apply SourceSafety.schedule_safe M src dst n B bounds
    · intro i hi
      exact hb.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
  · intro xs initial hin
    obtain ⟨final,he,hm⟩ := ScheduledSource.complete M src dst n B positive order schedule initial
    refine ⟨final,he,?_,?_⟩
    · intro o j _
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      have input := hin (0 : Fin 1) j trivial
      change (initial.mem src j.1.val).read .real=xs (0 : Fin 1) j at input
      change mapCell id (final.mem dst j.1.val)=Cell.mk .real (f (xs (0 : Fin 1) j))
      rw [hm]
      change mapCell id (if dst=dst ∧ j.1.val<n then
        Cell.mk .real (scalar M ((initial.mem src j.1.val).read .real))
        else initial.mem dst j.1.val)=Cell.mk .real (f (xs (0 : Fin 1) j))
      rw [if_pos ⟨rfl,j.1.isLt⟩]
      change Cell.mk .real (scalar M ((initial.mem src j.1.val).read .real))=_
      rw [identify,input]
    · intro r i outside
      have untouched : ¬(r=dst ∧ i<n) := by
        rintro ⟨rfl,hi⟩
        have h := outside.1 (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
        exact h.elim (fun h=>h rfl) (fun h=>h rfl)
      rw [hm]
      exact if_neg untouched

theorem real_implements (src dst : RegionName) (n B : Nat) (positive : 0<B)
    (order : List Nat) (schedule : order.Perm (List.range (grid n B))) :
    (io KernelArithmetic.realAlgebra src dst n B order).Implements
      (formula KernelArithmetic.mathematical src dst n) :=
  implements _ _ KernelArithmetic.scalar_real src dst n B positive order schedule

theorem native_implements (profile : FP.Config) (src dst : RegionName) (n B : Nat)
    (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) :
    (io (KernelArithmetic.nativeAlgebra profile) src dst n B order).Implements
      (formula (KernelArithmetic.native profile) src dst n) :=
  implements _ _ (KernelArithmetic.scalar_native profile) src dst n B positive order schedule

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.IOComplete
