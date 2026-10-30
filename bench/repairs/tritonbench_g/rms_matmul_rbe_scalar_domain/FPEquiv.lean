import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Pair

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
open VeriTile.Triton FP.Structural Pair
open scoped VeriTile.Spec
noncomputable section

/-- Existing output-row index commutation on the actual typed Q/K/V source.
The complete array IO, arbitrary launch order, runtime EPS/THETA, both weight
formats and separate FP16/E5 outputs are preserved. Floating casts, dot/reduction
order and separate multiply/add operations are unchanged. The numerical model
is the original scalar/instruction interpretation, not an extra compiler claim. -/
specification rms_qkv_index_commuted_equiv {α : Type} [Inhabited α]
    (a : Arithmetic α) (zero : ZeroLaws a) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    original a fp8 p epsilon theta order ≡[[]] indexCommuted a fp8 p epsilon theta order :=
  ArrayProgramIO.Implementation.of_equivalentOn (variant a fp8 p epsilon theta order)
    (equivalent a zero fp8 p epsilon theta order even apart)

#axiomsClean rms_qkv_index_commuted_equiv
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
