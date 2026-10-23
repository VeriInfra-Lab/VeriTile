import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeBridge
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IORealForward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IORealBackward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeCorrect
open VeriTile.Triton FP.Structural KernelProgram Interface RuntimeTransport
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def bind (M : Algebra α) (scale : ℝ) (u : State α) :=
  u.setReg "scale" .real [] (fun _ => RuntimeIO.scaleValue M scale)

 theorem initial (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (u : State α) :
    (RuntimeIO.invocation M kind Bool.false s f scale bh iv).initial u=
      (job M kind s f scale bh iv).initial (bind M scale u) := rfl

 theorem bound (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (u : State α) :
    Bound (RuntimeIO.scaleValue M scale) ((job M kind s f scale bh iv).initial (bind M scale u)) := by
  simp [Bound,job,Invocation.initial,bindRegisters,bind,State.setReg]

 theorem literal_schedule (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (vals : (RuntimeIO.literalIO M kind s f scale bh iv).Inputs) :
    (RuntimeIO.literalIO M kind s f scale bh iv).schedule vals=[job M kind s f scale bh iv] := by
  cases kind <;> rfl

/-- Transfer the established independent result to the actual runtime source.
The ABI binding is constructed and its run/safety link proved above. -/
 theorem transfer (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat)
    (domain : (RuntimeIO.literalIO M kind s f scale bh iv).Inputs → Prop)
    (formula : (RuntimeIO.literalIO M kind s f scale bh iv).Inputs →
      (RuntimeIO.literalIO M kind s f scale bh iv).Outputs)
    (old : (RuntimeIO.literalIO M kind s f scale bh iv).ImplementsOn domain formula) :
    (RuntimeIO.io M kind Bool.false s f scale bh iv).ImplementsOn domain formula := by
  intro vals hd bounds hb u hin
  have hbind : (RuntimeIO.literalIO M kind s f scale bh iv).InputBinding vals (bind M scale u) := hin
  obtain ⟨safe,v,ran,output,frame⟩ := old vals hd bounds hb (bind M scale u) hbind
  have pair := RuntimeBridge.literal_pair M kind s f scale
  have hs := bound M kind s f scale bh iv u
  refine ⟨?_,v,?_,output,frame⟩
  · rw [literal_schedule,scheduleSafe_singleton] at safe
    change ScheduleSafe bounds [RuntimeIO.invocation M kind Bool.false s f scale bh iv] u
    rw [scheduleSafe_singleton]
    change TraceSafe (FP.ProfiledRewrite.engine M) bounds (RuntimeIO.source kind Bool.false s f)
      ((RuntimeIO.invocation M kind Bool.false s f scale bh iv).initial u)
    rw [initial]
    exact (pair.safe bounds _ hs).mp safe
  · rw [literal_schedule,runSchedule_singleton] at ran
    change runSchedule [RuntimeIO.invocation M kind Bool.false s f scale bh iv] u=some v
    rw [runSchedule_singleton]
    change exec (FP.ProfiledRewrite.engine M) (RuntimeIO.source kind Bool.false s f)
      ((RuntimeIO.invocation M kind Bool.false s f scale bh iv).initial u)=some v
    rw [initial]
    exact (pair.executes _ hs).symm.trans ran

 theorem forward (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (RuntimeIO.io KernelReal.algebra .forward Bool.false s f scale bh iv).Implements
      (forwardFormula s f scale) :=
  transfer KernelReal.algebra .forward s f scale bh iv (fun _ => True) _
    (IORealForward.complete s f scale bh iv)

 theorem backward (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) :
    (RuntimeIO.io KernelReal.algebra .backward Bool.false s f scale bh iv).ImplementsOn
      (SavedInputs s f iv a g terminal) (backwardFormula s scale a g terminal) :=
  transfer KernelReal.algebra .backward s f scale bh iv _ _ (IORealBackward.complete s f scale bh iv a g terminal)

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeCorrect
