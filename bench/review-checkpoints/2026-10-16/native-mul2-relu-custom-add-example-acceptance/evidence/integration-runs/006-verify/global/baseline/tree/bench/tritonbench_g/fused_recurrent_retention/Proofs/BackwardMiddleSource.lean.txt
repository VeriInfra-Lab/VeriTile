import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardHeaderSource

/-! Exact middle section between the original backward loops: the barrier,
all six pointer rebases, and the zero d_h initialization. This section executes
even when T=0; no rebase or subtraction is deleted. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddleSource
open VeriTile.Triton
open ForwardHeaderSource (mulRef)
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def lastOffset (p : Layout) (stride : Nat) : Op .nat [] :=
  .mul .nat .nil (.sub .nat .nil (.constNat p.T) (.constNat 1)) (.constNat stride)

def pointerOp (p : Layout) (n headStride stride : Nat) (idx : RegName)
    (r : RegionName) : Op .ptr [n] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarR
      (.add .nat .scalarL (.add .nat .nil (mulRef "i_bh" headStride) (mulRef idx n)) (.arange n))
      (lastOffset p stride))

def gradientPointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BK] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarR
      (.add .nat .scalarL
        (.add .nat .nil
          (.mul .nat .nil (.add .nat .nil (.ref .nat [] "i_bh")
            (.mul .nat .nil (mulRef "i_v" p.B) (.constNat p.H))) (.constNat p.qHead))
          (mulRef "i_k" p.BK)) (.arange p.BK)) (lastOffset p p.DK))

def outputPointerOp (p : Layout) (r : RegionName) : Op .ptr [p.BV] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat .scalarR
      (.add .nat .scalarL
        (.add .nat .nil
          (.mul .nat .nil (.add .nat .nil (.ref .nat [] "i_bh")
            (.mul .nat .nil (mulRef "i_k" p.B) (.constNat p.H))) (.constNat p.vHead))
          (mulRef "i_v" p.BV)) (.arange p.BV)) (lastOffset p p.DV))

def code (p : Layout) (r : BackwardRegions) : List Stmt :=
  [.ifThen (.constBool Bool.false) [],
   .assign .ptr [p.BK] "p_q" (pointerOp p p.BK p.qHead p.DK "i_k" r.q),
   .assign .ptr [p.BK] "p_k" (pointerOp p p.BK p.qHead p.DK "i_k" r.k),
   .assign .ptr [p.BV] "p_do" (pointerOp p p.BV p.vHead p.DV "i_v" r.do_),
   .assign .ptr [p.BV] "p_v" (pointerOp p p.BV p.vHead p.DV "i_v" r.v),
   .assign .ptr [p.BK] "p_dk" (gradientPointerOp p r.dk),
   .assign .ptr [p.BV] "p_dv" (outputPointerOp p r.dv),
   .assign .real [p.BK,p.BV] "d_h" (.full [p.BK,p.BV] (.const 0))]

theorem middle_exact (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    BackwardSource.middleCode p r hasSeed swapped = code p r := by
  cases swapped
  all_goals simp only [BackwardSource.middleCode,ForwardSource.eraseCode,
    Program.backwardSecond,Program.backwardFirst,Program.backward,TypedSource.backward,
    TypedSource.backwardOriginal,TypedSource.backwardCandidate,Bool.false_eq_true,↓reduceIte,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,List.map_cons,List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,code,pointerOp,gradientPointerOp,
    outputPointerOp,lastOffset,mulRef]
  all_goals rfl

#axiomsClean middle_exact
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddleSource
