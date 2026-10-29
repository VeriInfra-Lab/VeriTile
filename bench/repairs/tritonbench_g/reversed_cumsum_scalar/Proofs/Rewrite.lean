import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPProgram
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedContext

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Rewrite
open VeriTile.Triton FP.Structural FPProgram
open VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedContext
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

/-- Only the two signed logical row/stride products in the input pointer are commuted. -/
def candidateBase (H : Nat) (strides : Strides) : Op .int [] :=
  .add .int .nil
    (.mul .int .nil (.castNatToInt (.constNat strides.batch))
      (.tritonDiv .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat H))))
    (.mul .int .nil (.castNatToInt (.constNat strides.head))
      (.tritonMod .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat H))))

def candidatePointer (h : Call) (c : Config) : Op .ptr [c.BT] :=
  .ptrAdd .scalarL (.ptrBase h.input)
    (.castIntToNat (.add .int .scalarL (candidateBase h.H h.inputStrides)
      (.mul .int .scalarR (.ref .int [c.BT] "times") (.castNatToInt (.constNat h.inputStrides.time)))))

theorem pointer_step {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (c : Config) (u : State α) :
    step M (.assign .ptr [c.BT] "p_s" (.alg (pointerOp h c h.input h.inputStrides))) u =
      step M (.assign .ptr [c.BT] "p_s" (.alg (candidatePointer h c))) u := by
  cases hi : u.regs .int [] "i_bh" <;>
    cases ht : u.regs .int [c.BT] "times" <;>
    simp [pointerOp,baseOp,candidatePointer,candidateBase,step,FP.Structural.evalExpr,
      evalOp_unfold,Op.tritonDiv,Op.tritonMod,hi,ht,numeric,numericLt,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      IntegerExpr.same_left,IntegerExpr.same_right,Int.mul_comm]

def candidateBody (h : Call) (c : Config) : List ComputeStmt :=
  (loopBody h c).take 5 ++
    .assign .ptr [c.BT] "p_s" (.alg (candidatePointer h c)) :: (loopBody h c).drop 6

theorem source_split (h : Call) (c : Config) :
    loopBody h c = (loopBody h c).take 5 ++
      .assign .ptr [c.BT] "p_s" (.alg (pointerOp h c h.input h.inputStrides)) ::
        (loopBody h c).drop 6 := rfl

theorem candidate_source (h : Call) (c : Config) :
    (Kernels.indexCommuted h c).surfaceBody = initCode ++
      [.forRangeInt "__rev_i_t" (loopStart h c) (.constInt 1) (.constNat 1) (candidateBody h c)] := rfl

theorem body_equiv {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (c : Config) (u : State α) :
    run M (loopBody h c) u = run M (candidateBody h c) u := by
  rw [source_split h c]
  exact run_insert M _ _ _ _ (pointer_step M h c) u

/-- Complete typed state and failure equality. Floating primitives are identical
on the two sides; no reduction, scan, cancellation or IEEE premise is used. -/
theorem whole_state {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (c : Config) (u : State α) :
    FP.Structural.exec M (Kernels.source h c) u =
      FP.Structural.exec M (Kernels.indexCommuted h c) u := by
  change run M (Kernels.source h c).surfaceBody u = run M (Kernels.indexCommuted h c).surfaceBody u
  rw [source_body,candidate_source,body,statement]
  exact run_insert M _ [] _ _
    (signed_loop M _ _ _ _ _ _ (body_equiv M h c)) u

#axiomsClean pointer_step
#axiomsClean whole_state
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Rewrite
