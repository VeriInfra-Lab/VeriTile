import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.PublicNative

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

namespace VeriTile.Bench.Repairs.MixedSparseAttention
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section

/-- Complete source-to-native observation under the existing primitive model.
The FP32 loads/computation/store, selected FP16/BF16 dot operands, runtime scale
and source order are unchanged. This introduces no floating rewrite law. -/
theorem kernel_native (I : Native.CompareISA (BitVec 32)) (rt : Native.Runtime)
    (c : Config) (sep : Separated c) (layout : Contiguous c)
    (hbm : 0<c.BM) (hbn : 0<c.BN) (work : List Program)
    (order : work.Perm (programs (grid c))) :
    (NativeIO.io rt I c work).ObservesOn (NativeIO.observe c I)
      (KernelIO.domain c) (NativeIO.formula rt I c hbm) :=
  NativeIO.observes rt I c sep layout hbm hbn work order

#stmtSurfaceSubset kernel_native ⊆
  [ArrayProgramIO.ObservesOn,NativeIO.io,NativeIO.observe,KernelIO.domain,NativeIO.formula,
   Native.CompareISA,Native.Runtime,Native.Expr,Native.instInhabitedExpr,Config,Config.BM,Config.BN,
   Separated,Contiguous,Program,programs,grid,List.Perm]
#specNonCircular NativeIO.formula avoiding
  [FP.Structural.exec,runSchedule,Whole.runPrograms,Source.finalValues,Source.update,State.mem]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MixedSparseAttention
