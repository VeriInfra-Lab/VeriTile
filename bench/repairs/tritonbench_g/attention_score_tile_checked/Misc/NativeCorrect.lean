import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOProof
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawNative

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv
open VeriTile Triton FP.Structural Reused Geometry Physical NativeInstructions
noncomputable section

/-- The authoritative FP32 source and complete existing launch schedule. -/
abbrev kernelIO := NativeInterface.programIO

/-- One output value is observed only through the chosen native ISA. -/
def observation (I : ISA (BitVec 32)) (c : Shape)
    (_o : Fin (NativeInterface.outputPorts c).length)
    (_j : TileIndex (NativeInterface.outputPorts c)[_o].shape) : Graph → BitVec 32 :=
  NativeInterface.observe I

/-- Pure observation of the independent source-ordered TF32/scalar graph.
All input words and the exact arithmetic/reduction order are preserved. -/
def formula (I : ISA (BitVec 32)) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (xs : NativeInterface.Inputs c) :
    ChannelValues (BitVec 32) (NativeInterface.outputPorts c) := fun o j =>
  mapValue (NativeInterface.observe I) (NativeInterface.outputPorts c)[o].dtype
    (NativeInterface.formula c offset size scale window complement xs o j)

/-- Full source-to-native observation with exact input cells and raw frame.
ISA primitives are model parameters; this is the established native result,
with no extra kernel or floating transformation law. -/
theorem kernel_native (I : ISA (BitVec 32)) (c : Shape) (abi : ABI c)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (_domain : Whole002.Representation.Domain c abi offset size window)
    (_tile : c.B*c.D≤1048576) :
    (kernelIO c offset size scale window complement).ObservesOnCells
      (observation I c) (NativeInterface.initialOutput c)
      (formula I c offset size scale window complement) :=
  ProgramIO.ObservesOnCells.of_implementsOnCells
    (NativeInterface.Proof.implements c offset size scale window complement)
    (observation I c)

#stmtSurfaceSubset kernel_native ⊆
  [ISA,Shape,ABI,Whole002.Representation.Domain,Shape.B,Shape.D,
   ProgramIO.ObservesOnCells,
   Graph,NativeStaging.Expr,NativeStaging.instInhabitedExpr,
   kernelIO,observation,NativeInterface.initialOutput,formula]
#specNonCircular formula avoiding [kernel,nativeExecute,FP.Structural.exec,FPGrid.run]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv
