import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOSafetyGrid

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOResults
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem implements (a : Arithmetic α) (zero : ZeroLaws a) (native fp8 : Bool)
    (p : Fin 3 → Arguments) (epsilon theta : α) (order : List (Nat → Nat))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    (IO.qkvIO a native fp8 p epsilon theta order).ImplementsOn (IO.InputDomain fp8 p)
      (IOOutput.expected a native fp8 p epsilon theta order) := by
  apply ArrayProgramIO.ImplementsOn.intro
  · intro vals domain capacity bounds initial binding
    exact IOSafety.schedule_safe a zero native fp8 p epsilon theta order vals domain even apart
      capacity bounds initial binding
  · intro vals domain initial binding
    exact IOOutput.correctness a zero native fp8 p epsilon theta order vals domain even apart initial binding

theorem real_implements (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat)) (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    (IO.qkvIO mathArithmetic Bool.false fp8 p epsilon.value theta.value order).ImplementsOn
      (IO.InputDomain fp8 p) (Target.mathematical fp8 p epsilon theta order) := by
  have equal : IOOutput.expected mathArithmetic Bool.false fp8 p epsilon.value theta.value order=
      Target.mathematical fp8 p epsilon theta order :=
    funext (IOOutput.expected_real fp8 p epsilon theta order)
  rw [←equal]
  exact implements mathArithmetic math_zeroLaws Bool.false fp8 p epsilon.value theta.value order even apart

theorem native_implements (instructions : Native.Primitives) (zero : Native.ZeroRealization instructions)
    (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : Native.Word) (order : List (Nat → Nat))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    (IO.qkvIO (Native.arithmetic instructions) Bool.true fp8 p epsilon theta order).ImplementsOn
      (IO.InputDomain fp8 p) (Target.native instructions fp8 p epsilon theta order) :=
  implements (Native.arithmetic instructions) (Native.zero_laws instructions zero)
    Bool.true fp8 p epsilon theta order even apart

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOResults
