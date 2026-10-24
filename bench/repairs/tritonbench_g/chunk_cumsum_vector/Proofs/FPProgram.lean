import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.IndexProgram
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPProgram
open VeriTile.Triton IndexProgram
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def initCode (h : Call) (c : Config) : List ComputeStmt :=
  [.assign .nat [] "i_s" (.alg (.programId 0)),
   .assign .nat [] "i_bh" (.alg (.programId 1)),
   .assign .nat [] "i_b" (.alg (.floorDiv .nat .nil (.ref .nat [] "i_bh") (.constNat h.H))),
   .assign .nat [] "i_h" (.alg (.mod .nat .nil (.ref .nat [] "i_bh") (.constNat h.H))),
   .assign .nat [c.BT] "o_i" (.alg (.arange c.BT)),
   .assign .real [c.BT,c.BT] "m_s" (.alg (triangularOp c)),
   .assign .real [BS] "b_z" (.compute (.full [BS] (.alg .fp32 (.const 0)))),
   .assign .nat [] "start_s" (.alg (startOp "i_s" BS)),
   .assign .nat [BS] "lanes_s" (.alg (.arange BS)),
   .assign .bool [BS] "active_s" (.alg (activeOp h.S BS "lanes_s" "start_s")),
   .assign .nat [BS] "columns" (.alg (coordinateOp BS "start_s" "active_s" "lanes_s"))]
def loadOp (h : Call) (c : Config) : Op .real [c.BT,BS] :=
  .castFloat h.inputDType .real (.load h.inputDType.toTileDType
    (.ptr (.ref .ptr [c.BT,BS] "p_s"))
    (.maskOther (.ref .bool [c.BT,BS] "active")
      (.broadcast (.castFloat .real h.inputDType (.const 0)) [c.BT,BS])))
def storeOp (h : Call) (c : Config) : ComputeStmt :=
  .store h.outputDType.toTileDType [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
    (.alg (.castFloat .real h.outputDType (.withPrecision .real .fp32 (.ref .real [c.BT,BS] "b_c"))))
    (.mask (.ref .bool [c.BT,BS] "active"))
def loopBody (h : Call) (c : Config) : List ComputeStmt :=
  [.assign .nat [] "start_t" (.alg (startOp "i_t" c.BT)),
   .assign .nat [c.BT] "lanes_t" (.alg (.arange c.BT)),
   .assign .bool [c.BT] "active_t" (.alg (activeOp h.T c.BT "lanes_t" "start_t")),
   .assign .nat [c.BT] "times" (.alg (coordinateOp c.BT "start_t" "active_t" "lanes_t")),
   .assign .bool [c.BT,BS] "active" (.alg (maskOp c)),
   .assign .ptr [c.BT,BS] "p_s" (.alg (pointerOp c h.input h.inputStrides)),
   .assign .ptr [c.BT,BS] "p_z" (.alg (pointerOp c h.output h.outputStrides)),
   .assign .real [c.BT,BS] "b_s" (.compute (.alg .fp32 (loadOp h c))),
   .assign .real [c.BT,BS] "b_c" (.compute (.alg .fp32 (outputOp c))),
   storeOp h c,
   .ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
     [.assign .real [BS] "b_z" (.compute (.alg .fp32 (carryOp c)))]]
def body (h : Call) (c : Config) : List ComputeStmt :=
  initCode h c++[.forRangeDyn "i_t" (.constNat 0) (countOp h c) (.constNat 1) (loopBody h c)]

theorem source_body (h : Call) (c : Config) : (Kernels.source h c).surfaceBody=body h c := by
  rfl
#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPProgram
