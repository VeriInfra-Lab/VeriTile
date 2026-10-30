import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16IOExecution

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Interface
open VeriTile Triton FP.Structural Full Whole FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned.Interface (readInputs)
noncomputable section
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

theorem outside_real (p : Parameters) (work : List Program) (v : (realIO p work).Inputs)
    (r : RegionName) (off : Nat) (outside : (realIO p work).Outside v r off) :
    OutsideLogical p r off := by
  by_cases different : r≠p.output
  · exact Or.inl different
  · right
    intro b hb h hh n hn d hd
    exact (outside.1 (0 : Fin 1) (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,PUnit.unit) trivial).resolve_left different

theorem outside_native (p : Parameters) (work : List Program) (v : (nativeIO p work).Inputs)
    (r : RegionName) (off : Nat) (outside : (nativeIO p work).Outside v r off) :
    OutsideLogical p r off := by
  by_cases different : r≠p.output
  · exact Or.inl different
  · right
    intro b hb h hh n hn d hd
    exact (outside.1 (0 : Fin 1) (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,PUnit.unit) trivial).resolve_left different

theorem real_result (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (work : List Program) (order : (programs (grid p)).Perm work)
    (vals : (realIO p work).Inputs) (before : State Real.R)
    (bound : (realIO p work).InputBinding vals before) :
    ∃ final,runSchedule ((realIO p work).schedule vals) before=some final ∧
      (realIO p work).OutputBinding vals (realFormula p work vals) final ∧
      (realIO p work).Frame vals before final := by
  obtain ⟨final,run,values,frame⟩ := RawKernel.complete Real.algebra Real.model p g separate before work order
  refine ⟨final,?_,?_,?_⟩
  · exact (jobs_run _ _ _ _ _).trans run
  · intro o j active
    have zero : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change mapCell (fun x : Real.R => x.unbotD 0) (final.mem p.output
      (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=
        Cell.mk .bf16 (Real.expected p (readInputs .bf16 p vals)
          j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt,
      real_finite p g work vals before bound _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt]
    rfl
  · intro r off outside
    exact frame r off (outside_real p work vals r off outside)

theorem native_result (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (_plan : RawKernel.CompilerPlan p)
    (work : List Program) (order : (programs (grid p)).Perm work)
    (vals : (nativeIO p work).Inputs) (before : State Native.Expr)
    (bound : (nativeIO p work).InputBinding vals before) :
    ∃ final,runSchedule ((nativeIO p work).schedule vals) before=some final ∧
      (nativeIO p work).ObservedOutputBinding (nativeObserver I p work) vals
        (nativeFormula I p g.selected work vals) final ∧ (nativeIO p work).Frame vals before final := by
  obtain ⟨final,run,values,frame⟩ := RawKernel.complete (Native.algebraFor p) (Native.modelFor p) p g separate before work order
  refine ⟨final,?_,?_,?_⟩
  · exact (jobs_run _ _ _ _ _).trans run
  · intro o j active
    have zero : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change mapCell (RawKernel.nativeDecode I j.2.2.2.1.val)
      (mapCell id (final.mem p.output (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val)))=_
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt,
      native_finite p g work vals before bound _ _ _ j.1.isLt j.2.1.isLt]
    rfl
  · intro r off outside
    exact frame r off (outside_native p work vals r off outside)

#axiomsClean real_result
#axiomsClean native_result
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Interface
