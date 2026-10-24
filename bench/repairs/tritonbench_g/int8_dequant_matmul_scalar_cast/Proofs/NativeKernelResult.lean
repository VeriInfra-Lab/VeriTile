import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.NativeKernelResult
noncomputable section

/-- Complete native fixed-leaf source execution and exact FP16 words under
the original INT8/INT32/FP32/FP16 instruction profile and split order. -/
theorem kernel_native (s : Shape) (c : Config) (b : Buffers)
    (admitted : Source.Admitted c) (fresh : b.Fresh) (profile : FixedISAProfile)
    (selected : Selection s c) (bias : Bool) (events : List (PhysicalCommit s c))
    (order : (events.map PhysicalCommit.logical).Perm (allCommits s c))
    (initial : Memory Word32 Word16) (zero : Zeroed (fpOps profile) s c b initial) :
    ∃ final,Quote.sourceGrid (Quote.nativeInteger profile s c selected)
        s c bias b Source.runtimeDivisor events initial=some final ∧
      (∀ cell : MatrixCell s,final (outputAddress s b cell)=
        .inr (.inr (wordReference (fpOps profile) s c b selected Source.runtimeDivisor bias
          (Schedule.ofPermutation s c (events.map PhysicalCommit.logical) order) initial cell))) ∧
      (∀ address : Address,(∀ cell : MatrixCell s,address≠outputAddress s b cell) →
        final address=initial address) := by
  let q := Schedule.ofPermutation s c (events.map PhysicalCommit.logical) order
  refine ⟨gridRun (fpOps profile) s c b selected Source.runtimeDivisor bias q.events initial,?_,?_,?_⟩
  · exact Quote.sourceGrid_eq _ _ _ _ _ _ selected rfl events initial
  · exact KernelInterface.output_word (fpOps profile) s c b fresh selected Source.runtimeDivisor bias q initial zero
  · exact fun address outside => grid_frame (fpOps profile) s c b selected Source.runtimeDivisor bias q.events initial address outside

#axiomsClean kernel_native
#specNonCircular wordReference avoiding [Quote.sourceGrid,gridRun]
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.NativeKernelResult
