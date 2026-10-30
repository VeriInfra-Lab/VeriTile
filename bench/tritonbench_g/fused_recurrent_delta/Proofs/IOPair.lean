import bench.tritonbench_g.fused_recurrent_delta.PairIO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeCorrect
import bench.tritonbench_g.fused_recurrent_delta.Proofs.SourcePair
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeCarrier
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOPair
open VeriTile.Triton FP.Structural KernelProgram Interface
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

 theorem jobs (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) :
    (RuntimeIO.invocation M kind Bool.false s f scale bh iv).run u=
      (RuntimeIO.invocation M kind Bool.true s f scale bh iv).run u ∧
    (∀ bounds, (RuntimeIO.invocation M kind Bool.false s f scale bh iv).Safe bounds u ↔
      (RuntimeIO.invocation M kind Bool.true s f scale bh iv).Safe bounds u) := by
  have start : (RuntimeIO.invocation M kind Bool.true s f scale bh iv).initial u=
      (RuntimeIO.invocation M kind Bool.false s f scale bh iv).initial u := rfl
  have pid : ((RuntimeIO.invocation M kind Bool.false s f scale bh iv).initial u).pids 1=0 := by
    simp [RuntimeIO.invocation,job,Invocation.initial,bindRegisters,State.setReg]
  constructor
  · unfold Invocation.run
    rw [start]
    exact SourcePair.executes _ kind s f _ pid
  · intro bounds
    unfold Invocation.Safe
    rw [start]
    exact SourcePair.safe _ kind s f _ pid bounds

 theorem complete (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (RuntimeIO.io M kind Bool.false s f scale bh iv).Equivalent (variant M kind s f scale bh iv) := by
  have existsFormula : ∃ formula,
      (RuntimeIO.literalIO M kind s f scale bh iv).Implements formula := by
    cases kind with
    | forward => exact ⟨_,IOResult.forward M s f scale bh iv⟩
    | backward => exact ⟨_,IOResult.backward M s f scale bh iv⟩
  obtain ⟨formula,hformula⟩ := existsFormula
  have impl := RuntimeCorrect.transfer M kind s f scale bh iv (fun _ => True) formula hformula
  intro vals domain bounds hb u hin
  obtain ⟨safe,v,ran,_,frame⟩ := impl vals domain bounds hb u hin
  have same := jobs M kind s f scale bh iv u
  refine ⟨safe,?_,v,v,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [RuntimeIO.invocation M kind Bool.true s f scale bh iv] u
    rw [scheduleSafe_singleton]
    exact (same.2 bounds).mp (by simpa only [RuntimeIO.io,scheduleSafe_singleton] using safe)
  · change runSchedule [RuntimeIO.invocation M kind Bool.true s f scale bh iv] u=some v
    rw [runSchedule_singleton]
    exact same.1.symm.trans (by simpa only [RuntimeIO.io,runSchedule_singleton] using ran)
  · intro o j active
    cases kind <;> rfl

 theorem native_scale (isa : KernelNativeCarrier.Instructions) (routing : KernelNativeCarrier.Routing)
    (scale : KernelNativeCarrier.Word) :
    RuntimeIO.scaleValue (KernelNativeCarrier.algebra isa routing scale) KernelNativeCarrier.scaleSlot=
      KernelNativeCarrier.Node.scale scale := by
  simp [RuntimeIO.scaleValue,FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
    KernelNativeCarrier.algebra]


end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOPair
