import bench.tritonbench_g.cross_entropy2.Proofs.IOBinding
import bench.tritonbench_g.cross_entropy2.Proofs.SourceResults
import bench.tritonbench_g.cross_entropy2.Proofs.KernelSafety
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.CrossEntropy2.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem backward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) :
    (IO.backward M p row block args id id).Implements
      (IO.backwardFormula p (Reference.backward M p args block)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial bound
    change ScheduleSafe bounds [IO.launch M p (backwardSyntax p) row block args] initial
    rw [scheduleSafe_singleton]
    change TraceSafe (KernelNative.engine M) bounds (backwardSyntax p)
      ((IO.launch M p (backwardSyntax p) row block args).initial initial)
    have ht : KernelNative.target p ((IO.launch M p (backwardSyntax p) row block args).initial initial)=
        vals (1 : Fin 4) PUnit.unit := by
      simpa [KernelNative.target,IO.backward,IO.backwardInputs,mapValue] using
        bound (1 : Fin 4) PUnit.unit (Or.inl (by simp))
    apply KernelSafety.backward M p _ args (IOBinding.arguments M p _ row block args initial) bounds
    · exact bounded.2.1 (1 : Fin 4) PUnit.unit (Or.inl (by simp))
    · intro active
      exact bounded.2.1 (3 : Fin 4) PUnit.unit (Or.inr (by simpa only [ht] using active))
    · intro lane active
      exact bounded.2.1 (0 : Fin 4) (⟨block*p.block+lane.val,active⟩,PUnit.unit) (Or.inl (by simp))
    · exact bounded.2.1 (2 : Fin 4) PUnit.unit (Or.inl (by simp))
    · intro lane active
      exact bounded.2.2.1 (0 : Fin 1) (lane,PUnit.unit) active
  · intro vals initial bound
    let start := (IO.launch M p (backwardSyntax p) row block args).initial initial
    let xs := IO.vector (vals (0 : Fin 4))
    let label := vals (1 : Fin 4) PUnit.unit
    have hx : ∀ c, c<p.cols →
        (start.mem p.logits (p.logitsBase+start.pids 0*p.logitsStride+c)).read .real=xs c := by
      intro c hc
      simpa [start,xs,IO.vector,hc,IO.backward,IO.backwardInputs,mapValue] using
        bound (0 : Fin 4) (⟨c,hc⟩,PUnit.unit) (Or.inl (by simp))
    have ht : KernelNative.target p start=label := by
      simpa [start,label,KernelNative.target,IO.backward,IO.backwardInputs,mapValue] using
        bound (1 : Fin 4) PUnit.unit (Or.inl (by simp))
    have hl : (start.mem p.lse (p.lseBase+start.pids 0)).read .real=vals (2 : Fin 4) PUnit.unit := by
      simpa [start,IO.backward,IO.backwardInputs,mapValue] using
        bound (2 : Fin 4) PUnit.unit (Or.inl (by simp))
    have hu : label≠p.ignored →
        (start.mem p.upstream (p.upstreamBase+start.pids 0*p.upstreamStride)).read .real=vals (3 : Fin 4) PUnit.unit := by
      intro active
      simpa [start,IO.backward,IO.backwardInputs,mapValue] using
        bound (3 : Fin 4) PUnit.unit (Or.inr active)
    obtain ⟨final,ran,value,frame⟩ := SourceResults.backward M p start args
      (IOBinding.arguments M p _ row block args initial) xs label _ _ hx ht hl hu
    refine ⟨final,?_,?_,?_⟩
    · simpa only [IO.backward,runSchedule_singleton,Invocation.run,IO.launch,KernelNative.engine] using ran
    · intro port lane active
      fin_cases port
      change mapCell id (final.mem p.gradient
          (p.gradientBase+row*p.gradientStride+(block*p.block+lane.1.val)))=
        .mk .real (Reference.backward M p args block xs label
          (vals (2 : Fin 4) PUnit.unit) (vals (3 : Fin 4) PUnit.unit) lane.1)
      have result := value lane.1 active
      simp only [KernelNative.gradientOffset,start,IOBinding.row,IOBinding.block] at result
      rw [result]
      rfl
    · intro r o outside
      exact frame r o (fun lane active => outside.1 (0 : Fin 1) (lane,PUnit.unit) active)

#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy2.PublicIO
