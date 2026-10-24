import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PublicNative

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

/-- The unchanged FP32 source and complete metadata-selected serial grid. -/
def nativeIO := NativeIO.io

def nativeObservation := NativeIO.observe

def nativeReference := NativeIO.formula

/-- Observe every active output of the complete source through the existing
ordered SM80 plan. WordISA supplies only individual primitive instructions;
no native-to-Real claim or floating transformation assumption is introduced. -/
theorem correct_fp32 (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    (nativeIO g r extent).ObservesOn (nativeObservation g r extent ops)
      (PublicDomain.native g r extent) (nativeReference g r extent ops) :=
  NativeIO.public_observes g r extent ops

#stmtSurfaceSubset correct_fp32 ⊆
  [nativeIO,nativeObservation,nativeReference,PublicDomain.native,ProgramIO.ObservesOn,
   Raw.Geometry,Regions,RegionBounds,Native.WordISA,Native.Expr,Native.instInhabitedExpr]
#specNonCircular nativeReference avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,State.mem]
#axiomsClean correct_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel
