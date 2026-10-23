import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.FPEquiv

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
open VeriTile.Triton FP.Structural Pair
open scoped VeriTile.Spec
noncomputable section

/-- All-word instantiation of the same two typed sources. The distinct selected
native lowering's single-source results remain optional in Misc/NativeCorrect. -/
theorem native_word_instantiation (instructions : Native.Primitives)
    (zero : Native.ZeroRealization instructions) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : Native.Word) (order : List (Nat → Nat))
    (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT) :
    original (Native.arithmetic instructions) fp8 p epsilon theta order ≡[[]]
      indexCommuted (Native.arithmetic instructions) fp8 p epsilon theta order :=
  rms_qkv_index_commuted_equiv (Native.arithmetic instructions) (Native.zero_laws instructions zero)
    fp8 p epsilon theta order even apart

#axiomsClean native_word_instantiation
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.FPEquiv
