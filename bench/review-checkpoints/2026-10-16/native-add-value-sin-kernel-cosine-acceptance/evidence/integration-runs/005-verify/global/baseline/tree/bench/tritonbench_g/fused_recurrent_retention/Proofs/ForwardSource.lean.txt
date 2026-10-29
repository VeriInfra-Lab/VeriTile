import bench.tritonbench_g.fused_recurrent_retention.Proofs.Program
import VeriTile.Triton.Float.Erasure

/-! The exact real projection of the complete forward source. The loop body
is exposed as eleven source statements for execution proofs, with both masked
tails, the output reduction/store and every pointer advance retained. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSource
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def eraseCode (code : List ComputeStmt) : List Stmt :=
  (ComputeKernel.mk [] [] code).eraseDType.toAlgKernel.body

def kernel (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) : ComputeKernel :=
  (Program.forward swapped .fp32 .fp32 p r hasSeed storeFinal).eraseDType

def initialCode (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) : List Stmt :=
  eraseCode (Program.forwardParts swapped .fp32 .fp32 p r hasSeed storeFinal).before

def finalCode (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) : List Stmt :=
  eraseCode (Program.forwardParts swapped .fp32 .fp32 p r hasSeed storeFinal).after

def vectorLoad (n : Nat) (ptr mask : RegName) : Op .real [n] :=
  .load .real (.ptr (.ref .ptr [n] ptr))
    (.maskOther (.ref .bool [n] mask) (.broadcast (.const 0) [n]))

def stateUpdate (p : Layout) : Op .real [p.BV,p.BK] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarL (.ref .real [] "b_b") (.ref .real [p.BV,p.BK] "h"))
    (.mul .real (.consL (.consR .nil))
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BK] "_k"))
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BV] "_v")))

def outputProduct (p : Layout) : Op .real [p.BV,p.BK] :=
  .mul .real (.consR (.consSame .nil)) (.ref .real [p.BV,p.BK] "h")
    (.expandDim ⟨0,by simp⟩ (.ref .real [p.BK] "_q"))

def advance (n step : Nat) (ptr : RegName) : Stmt :=
  .assign .ptr [n] ptr (.ptrAdd .scalarR (.ref .ptr [n] ptr) (.constNat step))

def loopBody (p : Layout) : List Stmt :=
  [.assign .real [p.BK] "_k" (vectorLoad p.BK "p_k" "mask_bk"),
   .assign .real [p.BV] "_v" (vectorLoad p.BV "p_v" "mask_bv"),
   .assign .real [p.BK] "_q"
     (.mul .real .scalarR (vectorLoad p.BK "p_q" "mask_bk") (.const p.scale)),
   .assign .real [p.BV,p.BK] "h" (stateUpdate p),
   .assign .real [p.BV,p.BK] "_o" (outputProduct p),
   .assign .real [p.BV] "_o"
     (.reduceSum ⟨1,by simp⟩ Bool.false (.ref .real [p.BV,p.BK] "_o")),
   .store .real [p.BV] (.ptr (.ref .ptr [p.BV] "p_o"))
     (.ref .real [p.BV] "_o") (.mask (.ref .bool [p.BV] "mask_bv")),
   advance p.BK p.DK "p_q", advance p.BK p.DK "p_k",
   advance p.BV p.DV "p_o", advance p.BV p.DV "p_v"]

theorem loop_exact (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    eraseCode (Program.forwardParts swapped .fp32 .fp32 p r hasSeed storeFinal).body =
      loopBody p := by
  cases swapped
  all_goals simp only [eraseCode, Program.forwardParts, Program.forward,
    TypedSource.forward, TypedSource.forwardOriginal, TypedSource.forwardCandidate,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    loopBody, vectorLoad, stateUpdate, outputProduct, advance]
  all_goals rfl

theorem source_body (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    (kernel p r hasSeed storeFinal swapped).toAlgKernel.body =
      initialCode p r hasSeed storeFinal swapped ++
        [.forRange "_t" 0 p.T 1 (loopBody p)] ++
        finalCode p r hasSeed storeFinal swapped := by
  rw [← loop_exact p r hasSeed storeFinal swapped]
  cases swapped
  all_goals simp only [kernel, initialCode, finalCode, eraseCode,
    Program.forwardParts, Program.forward, TypedSource.forward,
    TypedSource.forwardOriginal, TypedSource.forwardCandidate,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, List.map_cons, List.map_nil, List.cons_append, List.nil_append]
  all_goals rfl

#axiomsClean loop_exact
#axiomsClean source_body
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSource
