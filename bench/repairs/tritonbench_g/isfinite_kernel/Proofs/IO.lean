import bench.repairs.tritonbench_g.isfinite_kernel.IO
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.AccessSafety
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Kernel

namespace VeriTile.Bench.Repairs.IsfiniteKernel.IOProof
open VeriTile.Triton FP.Structural
noncomputable section

/-- Kernel-level source result for logical classifier inputs. -/
theorem source_result {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A)
    (injective : ∀ i,i<n → ∀ j,j<n → output.address i=output.address j → i=j) :
    ∃ t,exec (Source.algebra precision classify) (source precision input output n B) s=some t ∧
      Kernel.Post classify input output n B s t := by
  obtain ⟨t,ran,memory⟩ := Source.executes precision classify input output n B s
  refine ⟨t,ran,?_⟩
  have post := Kernel.stores_post classify input output n B s injective
  simpa only [Kernel.Post,memory] using post

/-- The shared IO window bounds certify the actual source trace. -/
theorem safe {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat)
    (xs : (classificationIO precision classify input output n B pid).Inputs)
    (bounds : RegionName → Nat)
    (bounded : (classificationIO precision classify input output n B pid).InBounds bounds xs)
    (initial : State A) :
    ScheduleSafe bounds ((classificationIO precision classify input output n B pid).schedule xs) initial := by
  simp only [classificationIO,scheduleSafe_singleton]
  apply Source.access_safe
  · intro j hj
    exact bounded.2.1 (0 : Fin 1) (j,PUnit.unit) hj
  · intro j hj
    exact bounded.2.2.1 (0 : Fin 1) (j,PUnit.unit) hj

/-- Reuse the established execution, input-value and frame proof. -/
theorem implements {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat)
    (injective : ∀ i,i<n → ∀ j,j<n → output.address i=output.address j → i=j) :
    (classificationIO precision classify input output n B pid).Implements
      (classification precision classify input output n B pid) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial _
    exact safe precision classify input output n B pid xs bounds bounded initial
  · intro xs initial inputs
    let positioned : State A := {initial with pids := fun ax => if ax=0 then pid else 0}
    obtain ⟨t,ran,values,frame⟩ := source_result precision classify input output n B positioned injective
    dsimp [positioned] at values frame
    refine ⟨t,?_,?_,?_⟩
    · simpa only [classificationIO,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl,Option.getD,Bool.false_eq_true,↓reduceIte] using ran
    · intro o
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      intro j hj
      have pin := inputs (0 : Fin 1) j hj
      change (initial.mem "x" (input.address (pid*B+j.1.val))).read .real=xs (0 : Fin 1) j at pin
      change mapCell id (t.mem "y" (output.address (pid*B+j.1.val))) =
        Cell.mk .bool (classify (xs (0 : Fin 1) j))
      rw [values j.1 hj]
      change Cell.mk .bool (classify ((initial.mem "x" (input.address (pid*B+j.1.val))).read .real)) = _
      rw [pin]
    · intro r address outside
      apply frame r address
      by_cases hr : r="y"
      · right
        intro j hj
        exact (outside.1 (0 : Fin 1) (j,PUnit.unit) hj).resolve_left (not_not.mpr hr)
      · exact Or.inl hr

end
end VeriTile.Bench.Repairs.IsfiniteKernel.IOProof
