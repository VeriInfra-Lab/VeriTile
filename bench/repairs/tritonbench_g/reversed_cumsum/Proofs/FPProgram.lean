import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.IndexProgram
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPProgram
open VeriTile.Triton IndexProgram
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

def triOp (c : Config) : Op .real [c.BT,c.BT] :=
  .where (.le .nat (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .nat [c.BT] "o_i"))
    (.expandDim ⟨0,by simp⟩ (.ref .nat [c.BT] "o_i")))
    (.broadcast (.const 1.0) [c.BT,c.BT]) (.broadcast (.const 0.0) [c.BT,c.BT])

def initCode (c : Config) : List ComputeStmt :=
  [.ifThen (.constBool Bool.true)
     [.assign .int [] "i_s" (.alg (.castNatToInt (.programId 0))),
      .assign .int [] "i_bh" (.alg (.castNatToInt (.programId 1)))],
   .assign .nat [c.BT] "o_i" (.alg (.arange c.BT)),
   .assign .real [c.BT,c.BT] "m_s" (.alg (triOp c)),
   .assign .real [BS] "b_z" (.compute (.full [BS] (.alg .fp32 (.const 0))))]

def loadOp (h : Call) (c : Config) : Op .real [c.BT,BS] :=
  .castFloat h.inputDType .real (.load h.inputDType.toTileDType (.ptr (.ref .ptr [c.BT,BS] "p_s"))
    (.maskOther (.ref .bool [c.BT,BS] "active") (.broadcast (.castFloat .real h.inputDType (.const 0)) [c.BT,BS])))

def outputOp (c : Config) : Op .real [c.BT,BS] :=
  .add .real (.consL (.consSame .nil)) (.expandDim ⟨0,by simp⟩ (.ref .real [BS] "b_z"))
    (.dot (batch:=[]) (.ref .real [c.BT,c.BT] "m_s") (.ref .real [c.BT,BS] "b_s") {allowTF32:=some Bool.false})

def carryOp (c : Config) : Op .real [BS] :=
  .add .real (.consSame .nil) (.ref .real [BS] "b_z") (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT,BS] "b_s"))

def storeOp (h : Call) (c : Config) : ComputeStmt :=
  .store h.outputDType.toTileDType [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
    (.alg (.castFloat .real h.outputDType (.withPrecision .real .fp32 (.ref .real [c.BT,BS] "b_c"))))
    (.mask (.ref .bool [c.BT,BS] "active"))

def loopBody (h : Call) (c : Config) : List ComputeStmt :=
  [.assign .int [] "i_t" (.alg indexOp),
   .assign .int [] "start_t" (.alg (startTOp c)),
   .assign .int [] "start_s" (.alg startSOp),
   .assign .int [c.BT] "lane_t" (.alg (.castNatToInt (.arange c.BT))),
   .assign .int [BS] "lane_s" (.alg (.castNatToInt (.arange BS))),
   .assign .bool [c.BT] "active_t" (.alg (activeTOp h c)),
   .assign .bool [BS] "active_s" (.alg (activeSOp h)),
   .assign .int [c.BT] "times" (.alg (timeOp c)),
   .assign .int [BS] "features" (.alg featureOp),
   .assign .bool [c.BT,BS] "active" (.alg (maskOp c)),
   .assign .ptr [c.BT,BS] "p_s" (.alg (pointerOp h c h.input)),
   .assign .ptr [c.BT,BS] "p_z" (.alg (pointerOp h c h.output)),
   .assign .real [c.BT,BS] "b_s" (.compute (.alg .fp32 (loadOp h c))),
   .assign .real [c.BT,BS] "b_c" (.compute (.alg .fp32 (outputOp c))),
   storeOp h c,
   .ifThen nonnegativeOp [.assign .real [BS] "b_z" (.compute (.alg .fp32 (carryOp c)))]]

def statement (h : Call) (c : Config) : ComputeStmt :=
  .forRangeInt "__rev_i_t" (reverseStart h c) (.constInt 1) (.constNat 1) (loopBody h c)
def body (h : Call) (c : Config) := initCode c++[statement h c]

theorem source_body (h : Call) (c : Config) : (Kernels.source h c).surfaceBody=body h c := by rfl

#print axioms source_body
end VeriTile.Bench.Repairs.ReversedCumsum.FPProgram
