import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- The selected native Q→K→V grid implements its independent ordered word
reference, including final FP16/E5 casts and aliases. Existing scalar primitive
zero laws, K8 MMA, reduction order and rotary FMA choices remain explicit.
This is a source-to-native observation, not a new arithmetic rewrite. -/
theorem kernel_native (instructions : Native.Primitives)
    (zero : Native.ZeroRealization instructions) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : Native.Word) (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (order : List (Nat → Nat)) :
    (IO.qkvIO (Native.arithmetic instructions) Bool.true fp8 p epsilon theta order).ImplementsOn
      (IO.InputDomain fp8 p) (Target.native instructions fp8 p epsilon theta order) :=
  IOResults.native_implements instructions zero fp8 p epsilon theta order even apart

export Native (k8_port reduction_port)
#axiomsClean kernel_native
#axiomsClean k8_port
#axiomsClean reduction_port
#stmtSurfaceSubset kernel_native ⊆
  [Native.Primitives,Native.ZeroRealization,Native.Word,Native.arithmetic,
   Arguments,RegionName,TypedSource.Arguments.N,TypedSource.Arguments.X,TypedSource.Arguments.W,
   TypedSource.Arguments.RMS,TypedSource.Arguments.OUT,
   ArrayProgramIO.ImplementsOn,IO.qkvIO,IO.InputDomain,Target.native]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
