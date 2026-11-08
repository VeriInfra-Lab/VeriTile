import bench.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.Conv2d
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent grouped cross-correlation. Spatial coordinates are logical
output coordinates; inputArray supplies zero exactly outside the padded input.
The channel, height and width sums are the complete logical filter dimensions. -/
def groupedConvolution (s : Shape) (xs : ConvIO.LogicalInputs s) : ConvIO.LogicalOutputs s :=
  fun group batch row col feature => KernelNumeric.Value.scalar
    (∑ c ∈ Finset.range s.cg, ∑ h ∈ Finset.range s.kh, ∑ w ∈ Finset.range s.kw,
      xs.paddedInput s batch.val (group.val*s.cg+c)
        ((row.val : Int)*s.sh-s.ph+h) ((col.val : Int)*s.sw-s.pw+w) *
      xs.filter s (group.val*s.og+feature.val) c h w)

/-- The complete original grid implements grouped cross-correlation over the
finite strided FP32 tensor channels. All precision flags, the empty batch,
actual source access safety and the complete exterior frame are retained. -/
specification correct_real (s : Shape) (valid : Valid s) (flags : Flags) :
    Spec.Real (ConvIO.logicalRealIO s flags ⊨ groupedConvolution s) := by
  exact (ConvIO.logical_iff s valid flags).2 (ConvIO.real_implements s valid flags)

#stmtSurfaceSubset correct_real ⊆
  [Shape,Valid,Flags,Spec.Real,ConvIO.logicalRealIO,groupedConvolution,ConvIO.LogicalInputs,ConvIO.LogicalOutputs,
   ConvIO.Payload,KernelNumeric.Value,KernelNumeric.instInhabitedValue,LogicalProgramIO.Implements]
#specNonCircular ConvIO.mathematical avoiding
  [KernelSource.kernel,KernelGrid.launch,FP.Structural.exec,KernelGrid.value]
#axiomsClean correct_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Conv2d
