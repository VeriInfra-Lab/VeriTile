import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Public

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 7000000

theorem real_reference (p : Core.Parameters) (b : Buffers) (entry : Memory ℝ)
    (sms : 0<p.problem.SMS) (row column : Nat) (hr : row<p.problem.M) (hc : column<p.problem.N) :
    Targets.result Real.arithmetic p.problem b entry row column=
      RealMath.formula p.problem.K (Real.inputs b entry) row column := by
  obtain ⟨pid,hp,ordinal,ho,i,ri,ci⟩ := Core.covers p sms (⟨row,hr⟩,⟨column,hc⟩)
  have active : Source.active p.problem (p.problem.tile pid ordinal) (i.1,i.2.1) := by
    exact ⟨ri ▸ hr,ci ▸ hc⟩
  simpa only [ri,ci] using Targets.real p.problem b entry (p.problem.tile pid ordinal) (i.1,i.2.1) active

/-- Real theorem for the complete checked-wrapper branch strategy. It returns
the actual allocated Tensor, independent full K-sum, and entire storage frame. -/
theorem correct (p : Core.Parameters) (args : TensorBinding.Arguments)
    (initial : State ℝ) (entry : Memory ℝ) (layout : Layout p.problem args.descriptors)
    (strides : Core.Strides p args.descriptors) (bindings : Physical.Bindings p args.descriptors initial entry)
    (sms : 0<p.problem.SMS) :
    ∃ after,Public.invoke Real.arithmetic p args initial entry=some (args.output,after) ∧
      (∀ row<p.problem.M,∀ column<p.problem.N,
        TensorBinding.logicalValue after args.output (row*p.problem.N+column)=
          RealMath.formula p.problem.K (Real.inputs args.descriptors entry) row column) ∧
      TensorBinding.PhysicalFrame p.problem args.descriptors entry after ∧
      (∀ i,after (args.a.storage,i)=entry (args.a.storage,i)) ∧
      (∀ i,after (args.b.storage,i)=entry (args.b.storage,i)) := by
  obtain ⟨after,ran,values,frame,inputA,inputB⟩ := Public.complete Real.arithmetic p args initial entry
    layout strides bindings rfl sms
  refine ⟨after,ran,?_,frame,inputA,inputB⟩
  intro row hr column hc
  rw [values row hr column hc,real_reference p args.descriptors entry sms row column hr hc]

#axiomsClean correct
#specNonCircular RealMath.formula avoiding [Public.invoke,Core.executeGrid,Source.run]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
noncomputable section
set_option maxHeartbeats 7000000

/-- All raw words are admitted. The only numerical ports are individual
scalar conversions and named fixed-size ISA instructions with incoming acc. -/
theorem fp_equiv (p : Core.Parameters) (args : TensorBinding.Arguments)
    (initial : State Word) (entry : Memory Word) (layout : Layout p.problem args.descriptors)
    (strides : Core.Strides p args.descriptors) (bindings : Physical.Bindings p args.descriptors initial entry)
    (sms : 0<p.problem.SMS) (plan : DotLayout p.problem.profile.blockK)
    (native : Numerics) (reference : Native.Reference)
    (ports : LocalPorts native reference.numerics Native.allWords) :
    ∃ after,Public.invoke (Native.arithmetic p.problem.profile plan native) p args initial entry=
        some (args.output,after) ∧
      (∀ row<p.problem.M,∀ column<p.problem.N,
        TensorBinding.logicalValue after args.output (row*p.problem.N+column)=
          Targets.result (Native.arithmetic p.problem.profile plan reference.numerics)
            p.problem args.descriptors entry row column) ∧
      TensorBinding.PhysicalFrame p.problem args.descriptors entry after ∧
      (∀ i,after (args.a.storage,i)=entry (args.a.storage,i)) ∧
      (∀ i,after (args.b.storage,i)=entry (args.b.storage,i)) := by
  obtain ⟨after,ran,values,frame,inputA,inputB⟩ := Public.complete
    (Native.arithmetic p.problem.profile plan native) p args initial entry layout strides bindings
    (Native.padding_zero p.problem.profile plan native reference ports) sms
  refine ⟨after,ran,?_,frame,inputA,inputB⟩
  intro row hr column hc
  rw [values row hr column hc]
  exact Native.target_equivalent p.problem args.descriptors entry plan native reference.numerics ports row column

#axiomsClean fp_equiv
#specNonCircular LocalPorts avoiding [Public.invoke,Core.executeGrid,Targets.result,Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBridge
