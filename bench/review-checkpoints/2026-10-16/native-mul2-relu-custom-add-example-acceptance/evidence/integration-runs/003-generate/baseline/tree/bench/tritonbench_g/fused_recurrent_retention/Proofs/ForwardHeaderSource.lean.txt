import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.StatePrimitives

/-! Exact source prologue and optional epilogue, including the T=0 final store.
These are literal AST decompositions; their execution proofs are separate. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardHeaderSource
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def mulRef (name : RegName) (n : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] name) (.constNat n)

def decayOp : Op .real [] :=
  .sub .real .nil (.const 1) (.exp2
    (.sub .real .nil (.const (-5))
      (.mul .real .nil (.natToReal (.ref .nat [] "i_h")) (.const 1.0))))

def pointerOp (n headStride : Nat) (idx : RegName) (r : RegionName) : Op .ptr [n] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarL (.add .nat .nil (mulRef "i_bh" headStride) (mulRef idx n)) (.arange n))

def outputPointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BV] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarL
      (.add .nat .nil
        (.mul .nat .nil
          (.add .nat .nil (.ref .nat [] "i_bh")
            (.mul .nat .nil (mulRef "i_k" p.B) (.constNat p.H)))
          (.constNat p.vHead))
        (mulRef "i_v" p.BV)) (.arange p.BV))

def keyMask (p : Layout) (swapped : Bool) : Op .bool [p.BK] :=
  let offs := .add .nat .scalarL (mulRef "i_k" p.BK) (.arange p.BK)
  if swapped then .gt .nat .scalarL (.constNat p.DK) offs
  else .lt .nat .scalarR offs (.constNat p.DK)

def valueMask (p : Layout) : Op .bool [p.BV] :=
  .lt .nat .scalarR (.add .nat .scalarL (mulRef "i_v" p.BV) (.arange p.BV)) (.constNat p.DV)

def maskKV (p : Layout) : Op .bool [p.BV,p.BK] :=
  .boolAnd (.consL (.consR .nil))
    (.expandDim ⟨0,by simp⟩ (.ref .bool [p.BK] "mask_bk"))
    (.expandDim ⟨1,by simp⟩ (.ref .bool [p.BV] "mask_bv"))

def statePointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BV,p.BK] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat (.consL (.consR .nil))
      (.add .nat .scalarL
        (.mul .nat .nil (mulRef "i_bh" p.DK) (.constNat p.DV))
        (.mul .nat .scalarR
          (.add .nat .scalarL (mulRef "i_k" p.BK) (.expandDim ⟨0,by simp⟩ (.arange p.BK)))
          (.constNat p.DV)))
      (.add .nat .scalarL (mulRef "i_v" p.BV) (.expandDim ⟨1,by simp⟩ (.arange p.BV))))

def header (p : Layout) (r : ForwardRegions) (swapped : Bool) : List Stmt :=
  [.assign .nat [] "i_v" (.programId 0),
   .assign .nat [] "i_k" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [] "i_h" (.tritonMod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)),
   .assign .real [] "b_b" decayOp,
   .assign .ptr [p.BK] "p_q" (pointerOp p.BK p.qHead "i_k" r.q),
   .assign .ptr [p.BK] "p_k" (pointerOp p.BK p.qHead "i_k" r.k),
   .assign .ptr [p.BV] "p_v" (pointerOp p.BV p.vHead "i_v" r.v),
   .assign .ptr [p.BV] "p_o" (outputPointerOp p r.o),
   .assign .bool [p.BK] "mask_bk" (keyMask p swapped),
   .assign .bool [p.BV] "mask_bv" (valueMask p),
   .assign .bool [p.BV,p.BK] "mask_kv" (maskKV p),
   .assign .real [p.BV,p.BK] "h" (.full [p.BV,p.BK] (.const 0))]

def seedCode (p : Layout) (r : ForwardRegions) (hasSeed : Bool) : List Stmt :=
  [.ifThen (.constBool hasSeed)
    [.assign .ptr [p.BV,p.BK] "p_init_s" (statePointerOp p r.initial),
     .assign .real [p.BV,p.BK] "h" (StatePrimitives.seedAdd p)]]

def finalCode (p : Layout) (r : ForwardRegions) (storeFinal : Bool) : List Stmt :=
  [.ifThen (.constBool storeFinal)
    [.assign .ptr [p.BV,p.BK] "p_final_s" (statePointerOp p r.final),
     .store .real [p.BV,p.BK] (.ptr (.ref .ptr [p.BV,p.BK] "p_final_s"))
       (.ref .real [p.BV,p.BK] "h") (.mask (.ref .bool [p.BV,p.BK] "mask_kv"))]]

theorem initial_exact (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    ForwardSource.initialCode p r hasSeed storeFinal swapped = header p r swapped ++ seedCode p r hasSeed := by
  cases swapped
  all_goals simp only [ForwardSource.initialCode, ForwardSource.eraseCode,
    Program.forwardParts, Program.forward, TypedSource.forward,
    TypedSource.forwardOriginal, TypedSource.forwardCandidate,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, List.map_cons, List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    header,seedCode,mulRef,decayOp,pointerOp,outputPointerOp,keyMask,valueMask,maskKV,
    statePointerOp,StatePrimitives.seedAdd,StatePrimitives.seedLoad,
    List.cons_append,List.nil_append,Bool.false_eq_true,↓reduceIte]
  all_goals rfl

theorem final_exact (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    ForwardSource.finalCode p r hasSeed storeFinal swapped = finalCode p r storeFinal := by
  cases swapped
  all_goals simp only [ForwardSource.finalCode, ForwardSource.eraseCode,
    Program.forwardParts, Program.forward, TypedSource.forward,
    TypedSource.forwardOriginal, TypedSource.forwardCandidate,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, Source.dynamicRange?,
    Option.map, Option.getD, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, List.map_cons, List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    finalCode,statePointerOp,mulRef]
  all_goals rfl

#axiomsClean initial_exact
#axiomsClean final_exact
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardHeaderSource
