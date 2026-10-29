import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Geometry
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealProgram

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 2400000
set_option maxRecDepth 24000

def initCode : List Stmt :=
  [.assign .int [] "i_bh" (.castNatToInt (.programId 0)),
   .assign .real [] "b_z" (.full [] (.const 0))]
abbrev numChunks := Geometry.chunks

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
def loadOp (c : Config) : Op .real [c.BT] :=
  .load .real (.ptr (.ref .ptr [c.BT] "p_s"))
    (.maskOther (.ref .bool [c.BT] "active") (.broadcast (.const 0) [c.BT]))

def loopBody (h : Call) (c : Config) : List Stmt :=
  [.assign .int [] "i_t" (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")),
   .assign .int [] "start_t" (startOp c),
   .assign .nat [c.BT] "lanes" (.arange c.BT),
   .assign .bool [c.BT] "active" (activeOp h c),
   .assign .int [c.BT] "times" (timeOp c),
   .assign .ptr [c.BT] "p_s" (pointerOp h c h.input h.inputStrides),
   .assign .ptr [c.BT] "p_o" (pointerOp h c h.output h.outputStrides),
   .assign .real [c.BT] "b_s" (loadOp c),
   .assign .real [] "b_zz" (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT] "b_s")),
   .assign .real [] "b_z" (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz")),
   .assign .real [c.BT] "b_o" (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealValues.outputOp c.BT),
   .store .real [c.BT] (.ptr (.ref .ptr [c.BT] "p_o"))
     (.ref .real [c.BT] "b_o") (.mask (.ref .bool [c.BT] "active"))]

def startIndexOp (h : Call) (c : Config) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (.constNat (Geometry.chunks h c))) (.constInt 1))

theorem startIndex_eval (h : Call) (c : Config) (u : BlockState) :
    evalOp (startIndexOp h c) u=some (Tile.scalar (1-(numChunks h c:Int))) := by
  simp [-Tile.scalar_eta,startIndexOp,evalOp,Tile.scalar,Tile.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub,zero_sub,neg_sub]

def statement (h : Call) (c : Config) : Stmt :=
  .forRangeInt "__rev_i_t"
    (startIndexOp h c)
    (.constInt 1) (.constNat 1) (loopBody h c)
def body (h : Call) (c : Config) := initCode ++ [statement h c]

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

#print axioms source_body
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealProgram
