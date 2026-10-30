import bench.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafety
import bench.tritonbench_g.attention_kernel_aligned.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
open VeriTile Triton FP.Structural Full Whole FPValue
noncomputable section
set_option maxHeartbeats 1000000

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (p : Parameters) (g : Guard p) (bounds : RegionName → Nat)
    (inputs : ∀ i : Point p,pointAddress p i<bounds (inputRegion p i.role))
    (outputs : ∀ b : Fin p.batches,∀ h : Fin p.heads,∀ n : Fin p.length,∀ d : Fin p.head,
      offset (oStrides p) b.val h.val n.val d.val<bounds p.output)
    (work : List Program) (legal : ∀ i∈work,i∈programs (grid p)) (before : State α) :
    ScheduleSafe bounds (jobs M P p work) before := by
  induction work generalizing before with
  | nil => trivial
  | cons i work ih =>
    refine ⟨?_,fun next _ => ih (fun j hj => legal j (by simp [hj])) next⟩
    change TraceSafe (backend M P) bounds (Source.source p) ((job M P p i).initial before)
    rw [job_initial]
    apply SourceSafety.source_safe M P p bounds _ (g.selected.blockN_pos p)
    apply SourceSafety.accesses p g bounds _ _ inputs outputs
    have member := (mem_programs (grid p) i.1 i.2.1 i.2.2).mp (legal i (by simp))
    exact member.2.1

theorem real_safe (p : Parameters) (g : Guard p) (work : List Program)
    (order : (programs (grid p)).Perm work) (vals : (realIO p work).Inputs)
    (bounds : RegionName → Nat) (bounded : (realIO p work).InBounds bounds vals)
    (before : State Real.R) : ScheduleSafe bounds ((realIO p work).schedule vals) before := by
  apply schedule_safe Real.algebra Real.model p g bounds
    (point_bound .fp16 p bounds (fun i j => bounded.2.1 i j trivial))
    (fun b h n d => bounded.2.2.1 (0 : Fin 1) (b,h,n,d,PUnit.unit) trivial)
    work (fun i hi => order.mem_iff.mpr hi) before

theorem native_safe (p : Parameters) (g : Guard p) (work : List Program)
    (order : (programs (grid p)).Perm work) (vals : (nativeIO p work).Inputs)
    (bounds : RegionName → Nat) (bounded : (nativeIO p work).InBounds bounds vals)
    (before : State Native.Expr) : ScheduleSafe bounds ((nativeIO p work).schedule vals) before := by
  apply schedule_safe (Native.algebraFor p) (Native.modelFor p) p g bounds
    (point_bound .fp16 p bounds (fun i j => bounded.2.1 i j trivial))
    (fun b h n d => bounded.2.2.1 (0 : Fin 1) (b,h,n,d,PUnit.unit) trivial)
    work (fun i hi => order.mem_iff.mpr hi) before

theorem real_implements (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (realIO p work).Implements (realFormula p work) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded before _
    exact real_safe p g work order vals bounds bounded before
  · exact real_result p g separate work order

theorem native_observes (I : Native.Instructions) (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (plan : RawKernel.CompilerPlan p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (nativeIO p work).Observes (nativeObserver I p work) (nativeFormula I p g.selected work) := by
  intro vals _ bounds bounded before bound
  exact ⟨native_safe p g work order vals bounds bounded before,
    native_result I p g separate plan work order vals before bound⟩

#axiomsClean real_implements
#axiomsClean native_observes
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Interface
