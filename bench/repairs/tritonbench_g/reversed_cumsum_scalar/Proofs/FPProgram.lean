import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Geometry
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPProgram
open VeriTile.Triton
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

def initCode : List ComputeStmt :=
  [.assign .int [] "i_bh" (.alg (.castNatToInt (.programId 0))),
   .assign .real [] "b_z" (.compute (.full [] (.alg .fp32 (.const 0))))]
def startOp (c : Config) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_t") (.constInt c.BT)
def activeOp (h : Call) (c : Config) : Op .bool [c.BT] :=
  .lt .int .scalarR (.castNatToInt (.ref .nat [c.BT] "lanes"))
    (.sub .int .nil (.castNatToInt (.constNat h.T)) (.ref .int [] "start_t"))
def timeOp (c : Config) : Op .int [c.BT] :=
  .add .int .scalarL (.ref .int [] "start_t")
    (.castNatToInt (.where (.ref .bool [c.BT] "active") (.ref .nat [c.BT] "lanes")
      (.broadcast (.constNat 0) [c.BT])))
def baseOp (H : Nat) (strides : Strides) : Op .int [] :=
  .add .int .nil
    (.mul .int .nil (.tritonDiv .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat H)))
      (.castNatToInt (.constNat strides.batch)))
    (.mul .int .nil (.tritonMod .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat H)))
      (.castNatToInt (.constNat strides.head)))
def pointerOp (h : Call) (c : Config) (r : RegionName) (strides : Strides) : Op .ptr [c.BT] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.castIntToNat (.add .int .scalarL (baseOp h.H strides)
      (.mul .int .scalarR (.ref .int [c.BT] "times") (.castNatToInt (.constNat strides.time)))))
def loadOp (h : Call) (c : Config) : Op .real [c.BT] :=
  .castFloat h.inputDType .real (.load h.inputDType.toTileDType (.ptr (.ref .ptr [c.BT] "p_s"))
    (.maskOther (.ref .bool [c.BT] "active")
      (.broadcast (.castFloat .real h.inputDType (.const 0)) [c.BT])))
def outputOp (c : Config) : Op .real [c.BT] :=
  .add .real (.consR .nil)
    (.sub .real (.consSame .nil) (.ref .real [c.BT] "b_s")
      (.scan .sum ⟨0,by simp⟩ .forward (.ref .real [c.BT] "b_s")))
    (.expandDim ⟨0,by simp⟩ (.ref .real [] "b_z"))
def storeOp (h : Call) (c : Config) : ComputeStmt :=
  .store h.outputDType.toTileDType [c.BT] (.ptr (.ref .ptr [c.BT] "p_o"))
    (.alg (.castFloat .real h.outputDType (.withPrecision .real .fp32 (.ref .real [c.BT] "b_o"))))
    (.mask (.ref .bool [c.BT] "active"))
def loopBody (h : Call) (c : Config) : List ComputeStmt :=
  [.assign .int [] "i_t" (.alg (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t"))),
   .assign .int [] "start_t" (.alg (startOp c)),
   .assign .nat [c.BT] "lanes" (.alg (.arange c.BT)),
   .assign .bool [c.BT] "active" (.alg (activeOp h c)),
   .assign .int [c.BT] "times" (.alg (timeOp c)),
   .assign .ptr [c.BT] "p_s" (.alg (pointerOp h c h.input h.inputStrides)),
   .assign .ptr [c.BT] "p_o" (.alg (pointerOp h c h.output h.outputStrides)),
   .assign .real [c.BT] "b_s" (.compute (.alg .fp32 (loadOp h c))),
   .assign .real [] "b_zz" (.compute (.alg .fp32 (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT] "b_s")))),
   .assign .real [] "b_z" (.compute (.alg .fp32 (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz")))),
   .assign .real [c.BT] "b_o" (.compute (.alg .fp32 (outputOp c))),
   storeOp h c]
def countOp (h : Call) (c : Config) : Op .nat [] :=
  .constNat (Geometry.chunks h c)
def loopStart (h : Call) (c : Config) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (countOp h c)) (.constInt 1))
def statement (h : Call) (c : Config) : ComputeStmt :=
  .forRangeInt "__rev_i_t" (loopStart h c) (.constInt 1) (.constNat 1) (loopBody h c)
def body (h : Call) (c : Config) := initCode++[statement h c]

theorem source_body (h : Call) (c : Config) : (Kernels.source h c).surfaceBody=body h c := by
  rfl

#print axioms source_body
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPProgram
