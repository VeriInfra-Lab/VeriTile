import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumExecution

/-! Exact repaired cumulative source, including the exhausted-batch guard.
The unchanged triangular mask, load, dot and store expressions are reused. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000

def pointerOp (p : Layout) (r : RegionName) (swapped : Bool) : Op .blockPtr [p.BT,p.BS] :=
  .makeBlockPtrDyn r
    (.add .nat .nil
      (.add .nat .nil (baseOp p swapped)
        (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat p.BT)) (.constNat p.timeStride)))
      (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "i_s") (.constNat p.BS)) (.constNat p.channelStride)))
    [p.BT,p.BS] [p.BT,p.BS] [p.timeStride,p.channelStride] [0,0]

def decodeCode (p : Layout) : List Stmt :=
  [.assign .nat [] "i_s" (.mod .nat .nil (.ref .nat [] "linear") (.constNat (p.S/p.BS))),
   .assign .nat [] "i_t" (.mod .nat .nil
     (.div .nat .nil (.ref .nat [] "linear") (.constNat (p.S/p.BS))) (.constNat (p.T/p.BT))),
   .assign .nat [] "i_bh" (.div .nat .nil (.ref .nat [] "linear") (.constNat ((p.S/p.BS)*(p.T/p.BT))))]

def computeCode (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BS] "p_s" (pointerOp p s swapped),
   .assign .blockPtr [p.BT,p.BS] "p_o" (pointerOp p o swapped)] ++
    (RealCumProgram.computeCode p s o swapped).drop 2

def activeCode (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  decodeCode p ++ (RealCumProgram.initCode p).drop 3 ++ computeCode p s o swapped

def body (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool) : List Stmt :=
  [.assign .nat [] "linear" (.add .nat .nil (.programId 0) (.constNat base)),
   .ifThen (.lt .nat .nil (.ref .nat [] "linear") (.constNat (heads*(p.T/p.BT)*(p.S/p.BS))))
     (activeCode p s o swapped)]

def source (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) :=
  (if swapped then NativeSource.cumIndexFirst else NativeSource.cumOriginal)
    dtype out_dtype s o p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS heads base

theorem source_body (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.body =
      body p s o heads base swapped := by
  cases swapped <;> cases dtype <;> cases out_dtype
  all_goals simp only [source,Bool.false_eq_true,↓reduceIte,
    NativeSource.cumOriginal,NativeSource.cumIndexFirst,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumProgram
