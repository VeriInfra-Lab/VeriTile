import bench.repairs.tritonbench_g.cross_entropy2.Proofs.PublicBackward
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.SourceResults
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelTailSafety

namespace VeriTile.Bench.Repairs.CrossEntropy2.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) (positive : 0<p.block) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss) :
    (IO.forward M p row block args id id).Implements
      (IO.forwardFormula p (Reference.forward M p args block)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial bound
    change ScheduleSafe bounds [IO.launch M p (forwardSyntax p) row block args] initial
    rw [scheduleSafe_singleton]
    change TraceSafe (KernelNative.engine M) bounds (forwardSyntax p)
      ((IO.launch M p (forwardSyntax p) row block args).initial initial)
    apply KernelSafety.forward M p _ positive args (IOBinding.arguments M p _ row block args initial) bounds
    · exact bounded.2.1 (1 : Fin 2) PUnit.unit trivial
    · intro lane active
      exact bounded.2.1 (0 : Fin 2) (⟨block*p.block+lane.val,active⟩,PUnit.unit) trivial
    · intro owns
      have lower : (0 : Int)≤(block : Int)*p.block := mul_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
      have upper := lt_of_lt_of_le owns.2 (min_le_left _ _)
      have nonneg : 0≤KernelNative.target p ((IO.launch M p (forwardSyntax p) row block args).initial initial)-p.classStart :=
        le_trans lower owns.1
      have hk : (KernelNative.target p ((IO.launch M p (forwardSyntax p) row block args).initial initial)-p.classStart).toNat<p.cols := by
        exact_mod_cast (show ((KernelNative.target p ((IO.launch M p (forwardSyntax p) row block args).initial initial)-p.classStart).toNat : Int)<p.cols by
          simpa [Int.toNat_of_nonneg nonneg] using upper)
      exact bounded.2.1 (0 : Fin 2) (⟨_,hk⟩,PUnit.unit) trivial
    · exact bounded.2.2.1 (0 : Fin 3) PUnit.unit trivial
    · exact bounded.2.2.1 (1 : Fin 3) PUnit.unit trivial
    · exact bounded.2.2.1 (2 : Fin 3) PUnit.unit trivial
  · intro vals initial bound
    let start := (IO.launch M p (forwardSyntax p) row block args).initial initial
    let xs := IO.vector (vals (0 : Fin 2))
    let label := vals (1 : Fin 2) PUnit.unit
    have hx : ∀ c, c<p.cols →
        (start.mem p.logits (p.logitsBase+start.pids 0*p.logitsStride+c)).read .real=xs c := by
      intro c hc
      simpa [start,xs,IO.vector,hc,IO.forward,IO.forwardInputs,mapValue] using
        bound (0 : Fin 2) (⟨c,hc⟩,PUnit.unit) trivial
    have ht : KernelNative.target p start=label := by
      simpa [start,label,KernelNative.target,IO.forward,IO.forwardInputs,mapValue] using
        bound (1 : Fin 2) PUnit.unit trivial
    obtain ⟨final,ran,lse,loss,zloss,frame⟩ := SourceResults.forward M p start positive separate outputs args
      (IOBinding.arguments M p _ row block args initial) xs label hx ht
    simp only [start,IOBinding.row,IOBinding.block,IOBinding.memory] at lse loss zloss frame
    refine ⟨final,?_,?_,?_⟩
    · simpa only [IO.forward,runSchedule_singleton,Invocation.run,IO.launch,KernelNative.engine] using ran
    · intro port lane _
      fin_cases port
      · change mapCell id (final.mem p.lse (p.lseBase+block*p.rows+row))=
          .mk .real (Reference.lse M p args xs block)
        rw [lse]
        rfl
      · change mapCell id (final.mem p.loss (p.lossBase+block*p.rows+row))=
          .mk .real (Reference.loss M p args xs block label)
        rw [loss]
        rfl
      · change mapCell id (final.mem p.zLoss (p.zLossBase+block*p.rows+row))=
          .mk .real (Reference.zLoss M p args xs block label)
        rw [zloss]
        rfl
    · intro r o outside
      exact frame r o (outside.1 (1 : Fin 3) PUnit.unit trivial)
        (outside.1 (0 : Fin 3) PUnit.unit trivial) (outside.1 (2 : Fin 3) PUnit.unit trivial)


#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy2.PublicIO
