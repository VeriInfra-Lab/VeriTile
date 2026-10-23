import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.IndexProgram
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealMath

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealProgram
open VeriTile.Triton IndexProgram
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def initCode (h : Call) (c : Config) : List Stmt :=
  [.assign .nat [] "i_s" (.programId 0),
   .assign .nat [] "i_bh" (.programId 1),
   .assign .nat [] "i_b" (.floorDiv .nat .nil (.ref .nat [] "i_bh") (.constNat h.H)),
   .assign .nat [] "i_h" (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat h.H)),
   .assign .nat [c.BT] "o_i" (.arange c.BT),
   .assign .real [c.BT,c.BT] "m_s" (triangularOp c),
   .assign .real [BS] "b_z" (.full [BS] (.const 0)),
   .assign .nat [] "start_s" (startOp "i_s" BS),
   .assign .nat [BS] "lanes_s" (.arange BS),
   .assign .bool [BS] "active_s" (activeOp h.S BS "lanes_s" "start_s"),
   .assign .nat [BS] "columns" (coordinateOp BS "start_s" "active_s" "lanes_s")]
def loadOp (c : Config) : Op .real [c.BT,BS] :=
  .load .real (.ptr (.ref .ptr [c.BT,BS] "p_s"))
    (.maskOther (.ref .bool [c.BT,BS] "active") (.broadcast (.const 0) [c.BT,BS]))
def loopBody (h : Call) (c : Config) : List Stmt :=
  [.assign .nat [] "start_t" (startOp "i_t" c.BT),
   .assign .nat [c.BT] "lanes_t" (.arange c.BT),
   .assign .bool [c.BT] "active_t" (activeOp h.T c.BT "lanes_t" "start_t"),
   .assign .nat [c.BT] "times" (coordinateOp c.BT "start_t" "active_t" "lanes_t"),
   .assign .bool [c.BT,BS] "active" (maskOp c),
   .assign .ptr [c.BT,BS] "p_s" (pointerOp c h.input h.inputStrides),
   .assign .ptr [c.BT,BS] "p_z" (pointerOp c h.output h.outputStrides),
   .assign .real [c.BT,BS] "b_s" (loadOp c),
   .assign .real [c.BT,BS] "b_c" (outputOp c),
   .store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
     (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active")),
   .ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
     [.assign .real [BS] "b_z" (carryOp c)]]
def body (h : Call) (c : Config) : List Stmt :=
  initCode h c++[.forRangeDyn "i_t" (.constNat 0) (countOp h c) (.constNat 1) (loopBody h c)]

theorem source_body (h : Call) (c : Config) :
    (Kernels.source h c).eraseDType.toAlgKernel.body=body h c := by
  simp only [Kernels.source,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    Except.bind,bind,VeriTile.Triton.eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  cases h.inputDType <;> cases h.outputDType <;> rfl
#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealProgram
