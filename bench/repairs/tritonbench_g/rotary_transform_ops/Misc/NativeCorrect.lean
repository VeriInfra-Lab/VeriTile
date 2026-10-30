import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.IOComplete
import VeriTile.Meta.Specification

/-! Exact native output observation for the original complete FP32 grid.
This source has no optimization candidate. All payloads are admitted; the four
primitive callbacks and the existing selected FMA graph remain explicit. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps
open VeriTile.Triton FP.Structural Interface
noncomputable section

/-- `observeNative` exposes every output's exact raw word and forgets only
internal product provenance. Input Node values remain unrestricted. -/
theorem rotary_transform_ops_native_observation (c : Config) (valid : c.Valid)
    (p : Native.Instructions) (cumulative64 offsets64 : Bool) :
    (io (Native.ops p) c cumulative64 offsets64 observeNative).ImplementsOn
      (Ready c) (nativeReference c p) :=
  IOComplete.native_implements c valid p cumulative64 offsets64

#specNonCircular nativeReference avoiding
  [FP.Structural.exec,TypedGrid.run,ProgramModel.result,Native.graph,Native.pairWord,Native.ops]
#stmtSurfaceSubset rotary_transform_ops_native_observation ⊆
  [Config,Config.Valid,Native.Node,KernelEndpoints.instInhabitedNode,
    Native.Instructions,Native.ops,io,observeNative,
    ReadProgramIO.ImplementsOn,Ready,nativeReference]
#axiomsClean rotary_transform_ops_native_observation
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformOps
