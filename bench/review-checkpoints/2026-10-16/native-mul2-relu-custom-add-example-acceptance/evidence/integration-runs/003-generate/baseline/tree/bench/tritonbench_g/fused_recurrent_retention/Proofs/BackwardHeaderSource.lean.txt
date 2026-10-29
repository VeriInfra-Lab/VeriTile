import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardHeaderSource

/-! Exact backward initialization: all thirteen header statements and the
conditional mask/pointer/load/add seed branch. The transposed [BK,BV] layout
and source addition into zero are retained. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardHeaderSource
open VeriTile.Triton
open ForwardHeaderSource (mulRef decayOp pointerOp keyMask valueMask)
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def gradientPointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BK] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarL
      (.add .nat .nil
        (.mul .nat .nil (.add .nat .nil (.ref .nat [] "i_bh")
          (.mul .nat .nil (mulRef "i_v" p.B) (.constNat p.H))) (.constNat p.qHead))
        (mulRef "i_k" p.BK)) (.arange p.BK))

def maskKV (p : Layout) : Op .bool [p.BK,p.BV] :=
  .boolAnd (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .bool [p.BK] "mask_bk"))
    (.expandDim ⟨0,by simp⟩ (.ref .bool [p.BV] "mask_bv"))

def statePointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BK,p.BV] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL
        (.mul .nat .nil (mulRef "i_bh" p.DK) (.constNat p.DV))
        (.mul .nat .scalarR
          (.add .nat .scalarL (mulRef "i_k" p.BK) (.expandDim ⟨1,by simp⟩ (.arange p.BK)))
          (.constNat p.DV)))
      (.add .nat .scalarL (mulRef "i_v" p.BV) (.expandDim ⟨0,by simp⟩ (.arange p.BV))))

def seedLoad (p : Layout) : Op .real [p.BK,p.BV] :=
  .load .real (.ptr (.ref .ptr [p.BK,p.BV] "p_init_s"))
    (.maskOther (.ref .bool [p.BK,p.BV] "mask_kv") (.broadcast (.const 0) [p.BK,p.BV]))

def seedAdd (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "h") (seedLoad p)

def header (p : Layout) (r : BackwardRegions) (swapped : Bool) : List Stmt :=
  [.assign .nat [] "i_v" (.programId 0),
   .assign .nat [] "i_k" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [] "i_h" (.tritonMod .nat .nil (.ref .nat [] "i_bh") (.constNat p.H)),
   .assign .real [] "b_b" decayOp,
   .assign .ptr [p.BK] "p_q" (pointerOp p.BK p.qHead "i_k" r.q),
   .assign .ptr [p.BK] "p_k" (pointerOp p.BK p.qHead "i_k" r.k),
   .assign .ptr [p.BV] "p_v" (pointerOp p.BV p.vHead "i_v" r.v),
   .assign .ptr [p.BV] "p_do" (pointerOp p.BV p.vHead "i_v" r.do_),
   .assign .ptr [p.BK] "p_dq" (gradientPointerOp p r.dq),
   .assign .bool [p.BK] "mask_bk" (keyMask p swapped),
   .assign .bool [p.BV] "mask_bv" (valueMask p),
   .assign .real [p.BK,p.BV] "h" (.full [p.BK,p.BV] (.const 0))]

def seedCode (p : Layout) (r : BackwardRegions) (hasSeed : Bool) : List Stmt :=
  [.ifThen (.constBool hasSeed)
    [.assign .bool [p.BK,p.BV] "mask_kv" (maskKV p),
     .assign .ptr [p.BK,p.BV] "p_init_s" (statePointerOp p r.initial),
     .assign .real [p.BK,p.BV] "h" (seedAdd p)]]

theorem initial_exact (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    BackwardSource.initialCode p r hasSeed swapped = header p r swapped ++ seedCode p r hasSeed := by
  cases swapped
  all_goals simp only [BackwardSource.initialCode,ForwardSource.eraseCode,
    Program.backwardFirst,Program.backward,TypedSource.backward,TypedSource.backwardOriginal,
    TypedSource.backwardCandidate,Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Option.map,Option.getD,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,List.map_cons,List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,header,seedCode,mulRef,decayOp,pointerOp,
    gradientPointerOp,keyMask,valueMask,maskKV,statePointerOp,seedLoad,seedAdd,
    List.cons_append,List.nil_append,Bool.false_eq_true,↓reduceIte]
  all_goals rfl

#axiomsClean initial_exact
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardHeaderSource
