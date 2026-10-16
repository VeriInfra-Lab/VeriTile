import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardSource

/-! The complete backward source has two actual time loops. The first rebuilds
h from the initial seed and produces d_q; the second walks the rebased pointers
backwards, producing d_k/d_v with every pointer decrement retained. The barrier
remains in the source and its sequential algorithm projection is a no-op. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSource
open VeriTile.Triton
open ForwardSource (eraseCode vectorLoad advance)
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def kernel (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) : ComputeKernel :=
  (Program.backward swapped .fp32 .fp32 p r hasSeed).eraseDType

def initialCode (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) : List Stmt :=
  eraseCode (Program.backwardFirst swapped .fp32 .fp32 p r hasSeed).before

def middleCode (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) : List Stmt :=
  eraseCode (Program.backwardSecond swapped .fp32 .fp32 p r hasSeed).before

def stateUpdate (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil))
    (.mul .real .scalarL (.ref .real [] "b_b") (.ref .real [p.BK,p.BV] "h"))
    (.mul .real (.consR (.consL .nil))
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "_k"))
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "_v")))

def dqProduct (p : Layout) : Op .real [p.BK,p.BV] :=
  .mul .real (.consR (.consSame .nil)) (.ref .real [p.BK,p.BV] "h")
    (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "_do"))

def firstBody (p : Layout) : List Stmt :=
  [.assign .real [p.BK] "_k" (vectorLoad p.BK "p_k" "mask_bk"),
   .assign .real [p.BV] "_v" (vectorLoad p.BV "p_v" "mask_bv"),
   .assign .real [p.BV] "_do" (vectorLoad p.BV "p_do" "mask_bv"),
   .assign .real [p.BK,p.BV] "h" (stateUpdate p),
   .assign .real [p.BK,p.BV] "_d_q" (dqProduct p),
   .assign .real [p.BK] "d_q" (.mul .real .scalarR
     (.reduceSum ⟨1,by simp⟩ Bool.false (.ref .real [p.BK,p.BV] "_d_q")) (.const p.scale)),
   .store .real [p.BK] (.ptr (.ref .ptr [p.BK] "p_dq"))
     (.ref .real [p.BK] "d_q") (.mask (.ref .bool [p.BK] "mask_bk")),
   advance p.BK p.DK "p_k", advance p.BV p.DV "p_do",
   advance p.BV p.DV "p_v", advance p.BK p.DK "p_dq"]

def dStateUpdate (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BV] "d_h")
    (.mul .real (.consR (.consL .nil))
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "_q"))
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "_do")))

def dkOp (p : Layout) : Op .real [p.BK] :=
  .reduceSum ⟨1,by simp⟩ Bool.false
    (.mul .real (.consR (.consSame .nil)) (.ref .real [p.BK,p.BV] "d_h")
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BV] "_v")))

def dvOp (p : Layout) : Op .real [p.BV] :=
  .reduceSum ⟨0,by simp⟩ Bool.false
    (.mul .real (.consSame (.consR .nil)) (.ref .real [p.BK,p.BV] "d_h")
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "_k")))

def retreat (n stride : Nat) (ptr : RegName) : Stmt :=
  .assign .ptr [n] ptr (.ptrSub .scalarR (.ref .ptr [n] ptr) (.constNat stride))

def secondBody (p : Layout) : List Stmt :=
  [.assign .real [p.BV] "_do" (vectorLoad p.BV "p_do" "mask_bv"),
   .assign .real [p.BK] "_q" (.mul .real .scalarR (vectorLoad p.BK "p_q" "mask_bk") (.const p.scale)),
   .assign .real [p.BK] "_k" (vectorLoad p.BK "p_k" "mask_bk"),
   .assign .real [p.BV] "_v" (vectorLoad p.BV "p_v" "mask_bv"),
   .assign .real [p.BK,p.BV] "d_h" (dStateUpdate p),
   .assign .real [p.BK] "d_k" (dkOp p),
   .assign .real [p.BV] "d_v" (dvOp p),
   .assign .real [p.BK,p.BV] "d_h" (.mul .real .scalarR
     (.ref .real [p.BK,p.BV] "d_h") (.ref .real [] "b_b")),
   .store .real [p.BK] (.ptr (.ref .ptr [p.BK] "p_dk"))
     (.ref .real [p.BK] "d_k") (.mask (.ref .bool [p.BK] "mask_bk")),
   .store .real [p.BV] (.ptr (.ref .ptr [p.BV] "p_dv"))
     (.ref .real [p.BV] "d_v") (.mask (.ref .bool [p.BV] "mask_bv")),
   retreat p.BV p.DV "p_do", retreat p.BK p.DK "p_q", retreat p.BK p.DK "p_k",
   retreat p.BV p.DV "p_v", retreat p.BK p.DK "p_dk", retreat p.BV p.DV "p_dv"]

theorem first_exact (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    eraseCode (Program.backwardFirst swapped .fp32 .fp32 p r hasSeed).body = firstBody p := by
  cases swapped
  all_goals simp only [eraseCode,Program.backwardFirst,Program.backward,
    TypedSource.backward,TypedSource.backwardOriginal,TypedSource.backwardCandidate,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map,Option.getD,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,List.map_cons,List.map_nil,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,firstBody,vectorLoad,stateUpdate,dqProduct,advance]
  all_goals rfl

theorem second_exact (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    eraseCode (Program.backwardSecond swapped .fp32 .fp32 p r hasSeed).body = secondBody p := by
  cases swapped
  all_goals simp only [eraseCode,Program.backwardSecond,Program.backwardFirst,Program.backward,
    TypedSource.backward,TypedSource.backwardOriginal,TypedSource.backwardCandidate,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Option.map,Option.getD,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,List.map_cons,List.map_nil,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,secondBody,vectorLoad,dStateUpdate,dkOp,dvOp,retreat]
  all_goals rfl

theorem source_body (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    (kernel p r hasSeed swapped).toAlgKernel.body =
      initialCode p r hasSeed swapped ++ [.forRange "i" 0 p.T 1 (firstBody p)] ++
      middleCode p r hasSeed swapped ++ [.forRange "_t" 0 p.T 1 (secondBody p)] := by
  rw [←first_exact p r hasSeed swapped,←second_exact p r hasSeed swapped]
  cases swapped
  all_goals simp only [kernel,initialCode,middleCode,eraseCode,Program.backwardSecond,
    Program.backwardFirst,Program.backward,TypedSource.backward,TypedSource.backwardOriginal,
    TypedSource.backwardCandidate,Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Option.map,Option.getD,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,List.map_cons,List.map_nil,List.cons_append,List.nil_append]
  all_goals rfl

#axiomsClean first_exact
#axiomsClean second_exact
#axiomsClean source_body
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSource
