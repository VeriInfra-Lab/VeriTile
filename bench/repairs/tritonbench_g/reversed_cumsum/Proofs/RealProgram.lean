import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.IndexProgram
import bench.tritonbench_g.reversed_cumsum.Proofs.RealExecution

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealProgram
open VeriTile.Triton IndexProgram
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

def layout (h : Call) (c : Config) : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram.Layout :=
  ⟨h.T*h.S,h.S,1,h.T,h.S,c.BT,BS⟩
def launch (row feature : Nat) : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram.Launch := ⟨feature,row⟩
def triOp (c : Config) : Op .real [c.BT,c.BT] :=
  (.where (.le .nat (.consR (.consL .nil))
       (.expandDim ⟨1,by simp⟩ (.ref .nat [c.BT] "o_i"))
       (.expandDim ⟨0,by simp⟩ (.ref .nat [c.BT] "o_i")))
       (.broadcast (.const 1.0) [c.BT,c.BT]) (.broadcast (.const 0.0) [c.BT,c.BT]))

def initCode (_h : Call) (c : Config) : List Stmt :=
  [.ifThen (.constBool Bool.true)
     [.assign .int [] "i_s" (.castNatToInt (.programId 0)),
      .assign .int [] "i_bh" (.castNatToInt (.programId 1))],
   .assign .nat [c.BT] "o_i" (.arange c.BT),
   .assign .real [c.BT,c.BT] "m_s" (triOp c),
   .assign .real [BS] "b_z" (.full [BS] (.const 0))]

def loadOp (c : Config) : Op .real [c.BT,BS] :=
  .load .real (.ptr (.ref .ptr [c.BT,BS] "p_s"))
    (.maskOther (.ref .bool [c.BT,BS] "active") (.broadcast (.const 0) [c.BT,BS]))
def outputOp (c : Config) : Op .real [c.BT,BS] :=
  .add .real (.consL (.consSame .nil)) (.expandDim ⟨0,by simp⟩ (.ref .real [BS] "b_z"))
    (.dot (batch:=[]) (.ref .real [c.BT,c.BT] "m_s") (.ref .real [c.BT,BS] "b_s") {allowTF32:=some Bool.false})
def carryOp (c : Config) : Op .real [BS] :=
  .add .real (.consSame .nil) (.ref .real [BS] "b_z") (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT,BS] "b_s"))

def loopBody (h : Call) (c : Config) : List Stmt :=
  [.assign .int [] "i_t" indexOp,
   .assign .int [] "start_t" (startTOp c),
   .assign .int [] "start_s" startSOp,
   .assign .int [c.BT] "lane_t" (.castNatToInt (.arange c.BT)),
   .assign .int [BS] "lane_s" (.castNatToInt (.arange BS)),
   .assign .bool [c.BT] "active_t" (activeTOp h c),
   .assign .bool [BS] "active_s" (activeSOp h),
   .assign .int [c.BT] "times" (timeOp c),
   .assign .int [BS] "features" featureOp,
   .assign .bool [c.BT,BS] "active" (maskOp c),
   .assign .ptr [c.BT,BS] "p_s" (pointerOp h c h.input),
   .assign .ptr [c.BT,BS] "p_z" (pointerOp h c h.output),
   .assign .real [c.BT,BS] "b_s" (loadOp c),
   .assign .real [c.BT,BS] "b_c" (outputOp c),
   .store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z")) (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active")),
   .ifThen nonnegativeOp [.assign .real [BS] "b_z" (carryOp c)]]
def statement (h : Call) (c : Config) : Stmt :=
  .forRangeInt "__rev_i_t" (reverseStart h c) (.constInt 1) (.constNat 1) (loopBody h c)
def body (h : Call) (c : Config) := initCode h c ++ [statement h c]

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
end VeriTile.Bench.Repairs.ReversedCumsum.RealProgram
