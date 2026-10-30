import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativeConnection

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry separately compares the complete actual program pairs. -/

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Instructions Plan LocalConformance)
noncomputable section

/-- The original FP32 forward source implements its independent native graph
observation over all original dimensions and words. Padded windows, source
access safety, every CTA's materialization, and the exact exterior frame are
part of the shared IO judgment. LocalConformance is the existing scalar-add /
bounded fixed-K8 ISA boundary, not an arithmetic-transformation premise. -/
theorem forward_fp_equiv (d : Host.Dimensions)
    (actual selected : Instructions) (plan : Plan) (numeric : LocalConformance actual selected) :
    (NativeIO.forward actual plan d).ImplementsCells (NativeReference.forward selected plan d) :=
  NativeConnection.forward d actual selected plan numeric

/-- Both backward grids are retained in order; inter consumes the actual
materialized intra gradients. No native-to-Real equality or geometry
restriction is added to the original arbitrary-word native domain. -/
theorem backward_fp_equiv (d : Host.Dimensions)
    (actual selected : Instructions) (plan : Plan) (numeric : LocalConformance actual selected) :
    (NativeIO.backward actual plan d).ImplementsCells (NativeReference.backward selected plan d) :=
  NativeConnection.backward d actual selected plan numeric

#axiomsClean forward_fp_equiv
#axiomsClean backward_fp_equiv
#stmtSurfaceSubset forward_fp_equiv ⊆
  [Host.Dimensions,Instructions,Plan,LocalConformance,MaterializedProgramIO.ImplementsCells,
   NativeIO.forward,NativeReference.forward,Native.Expr,Native.instInhabitedExpr,
   VeriTile.Bench.Repairs.LightningAttention.Native.Word]
#stmtSurfaceSubset backward_fp_equiv ⊆
  [Host.Dimensions,Instructions,Plan,LocalConformance,MaterializedProgramIO.ImplementsCells,
   NativeIO.backward,NativeReference.backward,Native.Expr,Native.instInhabitedExpr,
   VeriTile.Bench.Repairs.LightningAttention.Native.Word]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
