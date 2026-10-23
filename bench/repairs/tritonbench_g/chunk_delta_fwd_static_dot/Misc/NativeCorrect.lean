import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.IOImplements
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Misc.RawNative

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.FPEquiv
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The identical complete source grid and IO layout. Inputs are arbitrary
FP32 words; observation uses the established staged graph and scalar ISA model. -/
def kernelIO (I : WordPrimitives) (p : Parameters) (schedule : List PID) : ProgramIO (BitVec 32) Expr :=
  Interface.io Native.algebra Native.dot p schedule Expr.bits (Reference.nativeObserve I)
abbrev formula := Reference.nativeFormula

/-- Source-to-native observation, preserving padded lanes, ordered scalar FMAs,
one-subblock carry fusion, optional states and every output/frame. Primitive
meanings and compiler lowering are model parameters; no floating rewrite or
new numerical assumption is introduced. -/
theorem chunk_delta_kernel_native (I : WordPrimitives) (p : Parameters)
    (shape : Geometry.ShapeValid p) (schedule : List PID) (order : schedule.Perm (grid p)) :
    kernelIO I p schedule ⊨ formula I p shape :=
  IOImplements.native I p shape schedule order

#stmtSurfaceSubset chunk_delta_kernel_native ⊆
  [WordPrimitives,Parameters,Geometry.ShapeValid,PID,List.Perm,grid,BitVec,Expr,
   ProgramIO.Implements,kernelIO,formula,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.instInhabitedExpr]
#specNonCircular formula avoiding [ChunkDeltaFwd.kernel,FP.Structural.exec,runGrid,gridMemory]
#axiomsClean chunk_delta_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.FPEquiv
