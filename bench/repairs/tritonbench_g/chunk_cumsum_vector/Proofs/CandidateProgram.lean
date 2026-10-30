import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPProgram

/-! Source-binding decomposition of the displayed sum-left candidate.
The retained prefix, masks, loads, output expression and stores are shared. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateProgram
open VeriTile.Triton IndexProgram FPProgram
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def sumLeftCarryOp (c : Config) : Op .real [BS] :=
  .add .real (.consSame .nil)
    (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT,BS] "b_s"))
    (.ref .real [BS] "b_z")

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
     [.assign .real [BS] "b_z" (.compute (.alg .fp32 (sumLeftCarryOp c)))]]
def body (h : Call) (c : Config) : List ComputeStmt :=
  initCode h c++[.forRangeDyn "i_t" (.constNat 0) (countOp h c) (.constNat 1) (loopBody h c)]

theorem source_body (h : Call) (c : Config) : (Kernels.sumLeft h c).surfaceBody=body h c := by
  rfl
#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateProgram
