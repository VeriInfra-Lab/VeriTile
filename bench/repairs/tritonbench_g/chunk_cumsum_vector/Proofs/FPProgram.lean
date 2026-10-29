import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.IndexProgram
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPProgram
open VeriTile.Triton IndexProgram
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def initCode (h : Call) (c : Config) : List ComputeStmt :=
  [.assign .int [] "i_s" (.alg (.castNatToInt (.programId 0))),
   .assign .int [] "i_bh" (.alg (.castNatToInt (.programId 1))),
   .ifThen (.constBool Bool.true)
     [.assign .int [] "i_b" (.alg (.tritonDiv .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat h.H)))),
      .assign .int [] "i_h" (.alg (.tritonMod .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat h.H))))],
   .assign .nat [c.BT] "o_i" (.alg (.arange c.BT)),
   .assign .real [c.BT,c.BT] "m_s" (.alg (triangularOp c)),
   .assign .real [BS] "b_z" (.compute (.full [BS] (.alg .fp32 (.const 0)))),
   .assign .int [] "start_s" (.alg (startOp "i_s" BS)),
   .assign .int [BS] "lanes_s" (.alg (.castNatToInt (.arange BS))),
   .assign .bool [BS] "active_s" (.alg (activeOp h.S BS "lanes_s" "start_s")),
   .assign .int [BS] "columns" (.alg (coordinateOp BS "start_s" "active_s" "lanes_s"))]
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
  [.assign .int [] "start_t" (.alg (timeStartOp c)),
   .assign .int [c.BT] "lanes_t" (.alg (.castNatToInt (.arange c.BT))),
   .assign .bool [c.BT] "active_t" (.alg (activeOp h.T c.BT "lanes_t" "start_t")),
   .assign .int [c.BT] "times" (.alg (coordinateOp c.BT "start_t" "active_t" "lanes_t")),
   .assign .bool [c.BT,BS] "active" (.alg (maskOp c)),
   .assign .ptr [c.BT,BS] "p_s" (.alg (pointerOp c h.input h.inputStrides)),
   .assign .ptr [c.BT,BS] "p_z" (.alg (pointerOp c h.output h.outputStrides)),
   .assign .real [c.BT,BS] "b_s" (.compute (.alg .fp32 (loadOp h c))),
   .assign .real [c.BT,BS] "b_c" (.compute (.alg .fp32 (outputOp c))),
   storeOp h c,
   .ifThen (.ge .int .nil (.ref .int [] "i_t") (.constInt 0))
     [.assign .real [BS] "b_z" (.compute (.alg .fp32 (carryOp c)))]]
def body (h : Call) (c : Config) : List ComputeStmt :=
  initCode h c++[.forRangeInt "i_t" (.constInt 0) (.castNatToInt (countOp h c)) (.constNat 1) (loopBody h c)]

theorem source_body (h : Call) (c : Config) : (Kernels.source h c).surfaceBody=body h c := by
  rfl
#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPProgram
