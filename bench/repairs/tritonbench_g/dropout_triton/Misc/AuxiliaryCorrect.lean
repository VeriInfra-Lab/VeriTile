import bench.repairs.tritonbench_g.dropout_triton.Correct

/-! Auxiliary Real correctness of the integer-index candidate. The shared
original IO descriptor and historical declaration names are retained. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.Correct
open VeriTile Triton
open scoped VeriTile.Triton.MixedMaskedKernelIO₂
noncomputable section

/-- The existing candidate changes only integer index multiplication order. -/
def indexCommutedIO (n : Nat) (p : ℝ) : MixedMaskedKernelIO₂ :=
  { dropoutIO n p with
    source := indexCommutedKernel n p
    projection := RegionProof.projection Bool.true n blockSize p }

theorem dropout_index_commuted_correctness (n : Nat) (p : ℝ) :
    Spec.Real (indexCommutedIO n p ⊨ fun (xs : Fin blockSize → ℝ) (keep : Fin blockSize → Int) j =>
      (if keep j≠0 then xs j/(1-p) else 0 : ℝ)) := by
  apply MixedMaskedKernelIO₂.Implements.intro
  · intro bounds s hx hk ho
    exact (RegionProof.candidate_trace n p bounds s).mp
      (RealEndpoints.masked_safe n p bounds s hx hk ho)
  · exact RealEndpoints.candidate_masked_run n p

#stmtSurfaceSubset dropout_index_commuted_correctness ⊆
  [Spec.Real, blockSize, indexCommutedIO, MixedMaskedKernelIO₂.Implements,
   MixedMaskedKernelIO₂.B1, MixedMaskedKernelIO₂.B2, MixedMaskedKernelIO₂.Bout,
   MixedMaskedKernelIO₂.type1, MixedMaskedKernelIO₂.type2, MixedMaskedKernelIO₂.outType,
   ChanTy.carrier]
#axiomsClean dropout_index_commuted_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Correct
