import bench.tritonbench_g.matmul_persistent_triton.Proofs.InputConnection
import bench.tritonbench_g.matmul_persistent_triton.Proofs.IOBounds
import bench.tritonbench_g.matmul_persistent_triton.Native

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000
variable {α : Type} [Inhabited α]

private theorem observe_id (c : Cell α) : mapCell id c=c := by
  cases c with | mk d x => cases d <;> rfl

/-- Reuse the established complete grid theorem; this only connects its
independent reference and frame to the shared IO judgment. -/
theorem implements_of_target (P : Source.Arithmetic α α) (p : Core.Parameters)
    (b : RawKernel.Buffers) (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero)
    (f : ChannelValues α (inputs p) → ChannelValues α (outputs p))
    (target : ∀ xs s, (io P p).InputBinding xs s → ∀ j : TileIndex [p.problem.M,p.problem.N],
      Core.pack p.problem.profile (RawKernel.reference P p (isolated b) (entry p b s) j.1.val j.2.1.val)=
        Cell.mk (Core.dtype p.problem.profile) (f xs 0 j)) :
    (io P p).Implements f := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    exact schedule_safe P p sms xs bounds hb initial
  · intro xs initial hin
    obtain ⟨final,ran,values,frame⟩ := Core.gridExecution P p ((isolated b).descriptors p) initial
      (entry p b initial) ((isolated_layout p b layout).descriptors p (isolated b))
      (RawKernel.strides p (isolated b))
      ((entry_inputs p b initial).descriptors p (isolated b) initial (entry p b initial)) zero sms
    refine ⟨final,?_,?_,?_⟩
    · simpa [io,Core.executeGrid,schedule_run] using ran
    · intro o j _
      fin_cases o
      change mapCell id (final.mem "C" (j.1.val*p.cm+j.2.1.val*p.cn))=_
      rw [observe_id]
      calc
        final.mem "C" (j.1.val*p.cm+j.2.1.val*p.cn) =
            Core.pack p.problem.profile (RawKernel.reference P p (isolated b) (entry p b initial) j.1.val j.2.1.val) :=
          by simpa [Core.logicalAddress,Core.referenceValue,RawKernel.reference,Nat.mul_comm] using
            values (j.1,j.2.1) trivial
        _ = _ := target xs initial hin j
    · intro r offset hout
      apply frame
      by_cases hr : r="C"
      · right
        intro x _
        have h := hout.1 0 (x.1,x.2,PUnit.unit) trivial
        change r≠"C" ∨ offset≠x.1.val*p.cm+x.2.val*p.cn at h
        simpa [Core.logicalAddress,hr,Nat.mul_comm] using h
      · exact .inl hr

theorem mathematical_connection (p : Core.Parameters) (b : RawKernel.Buffers)
    (sms : 0<p.problem.SMS) (xs : (io Real.arithmetic p).Inputs) (s : State ℝ)
    (hin : (io Real.arithmetic p).InputBinding xs s) (j : TileIndex [p.problem.M,p.problem.N]) :
    Core.pack p.problem.profile
      (RawKernel.reference Real.arithmetic p (isolated b) (entry p b s) j.1.val j.2.1.val)=
      Cell.mk (Core.dtype p.problem.profile) (mathematical p xs 0 j) := by
  change Core.pack p.problem.profile
      (RawKernel.reference Real.arithmetic p (isolated b) (entry p b s) j.1.val j.2.1.val)=
      Cell.mk (Core.dtype p.problem.profile) (mathematicalValue p xs j)
  rw [mathematicalValue,pack_value,RawKernel.real_reference p (isolated b) (entry p b s) sms
    j.1.val j.2.1.val j.1.isLt j.2.1.isLt]
  congr 1
  rw [RealMath.formula,Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  congr 1
  · simpa [RawKernel.realInputs,entry,isolated,Nat.add_assoc] using input_a Real.arithmetic p xs s hin j.1 k
  · simpa [RawKernel.realInputs,entry,isolated,Nat.add_assoc] using input_b Real.arithmetic p xs s hin k j.2.1

theorem real_correct (p : Core.Parameters) (b : RawKernel.Buffers)
    (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS) :
    (io Real.arithmetic p).Implements (mathematical p) :=
  implements_of_target Real.arithmetic p b layout sms rfl (mathematical p)
    (mathematical_connection p b sms)

theorem native_correct (p : Core.Parameters) (b : RawKernel.Buffers)
    (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS)
    (plan : DotLayout p.problem.profile.blockK) (native : Floating.Numerics) (reference : Native.Reference)
    (ports : Floating.LocalPorts native reference.numerics Native.allWords) :
    (io (Native.arithmetic p.problem.profile plan native) p).Implements
      (result (Native.arithmetic p.problem.profile plan reference.numerics) p) := by
  apply implements_of_target _ p b layout sms (Native.padding_zero _ plan native reference ports)
  intro xs s hin j
  rw [RawKernel.reference,Native.target_equivalent p.problem ((isolated b).descriptors p)
    (entry p b s) plan native reference.numerics ports]
  exact result_connection _ _ p b xs s hin j

#axiomsClean real_correct
#axiomsClean native_correct
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
