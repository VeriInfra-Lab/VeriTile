import bench.repairs.tritonbench_g.attn_fwd_triton.Kernels
import bench.tritonbench_g.attn_fwd_triton.Proofs.Real

/-! Exact fragments of the complete candidate. Arithmetic fragments unchanged
by the repair are references to the original helpers, not copied proofs. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealContract.Parameters
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def source (BM BN BD : Nat) (p : Parameters) (reciprocal : Bool) : ComputeKernel :=
  (if reciprocal then TypedSource.reciprocal else TypedSource.repaired)
    p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.stage p.validStage

def code (BM BN BD : Nat) (p : Parameters) (r : Bool) : List Stmt :=
  (source BM BN BD p r).eraseDType.toAlgKernel.body

abbrev keyName := VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.keyName
abbrev scaleName := VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.scaleName
abbrev valueName := VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.valueName
abbrev maskName := VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation.maskName

def tail (BN : Nat) : Op .bool [1,BN] :=
  .lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n"))
    (.sub .nat .nil (.ref .nat [] "n_ctx") (.ref .nat [] "start_n"))

def keyMask (BD BN D : Nat) : Op .bool [BD,BN] :=
  .boolAnd (.consL (.consR .nil)) (tail BN)
    (.expandDim ⟨1, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat D)))

def valueMask (BN BD D : Nat) : Op .bool [BN,BD] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BN] "offs_n"))
      (.sub .nat .nil (.ref .nat [] "n_ctx") (.ref .nat [] "start_n")))
    (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat D)))

def queryMask (BM BD D : Nat) : Op .bool [BM,BD] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.ref .nat [] "n_ctx"))
    (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat D)))

def loadPrefix (BD BN D : Nat) (second : Bool) : List Stmt :=
  [.assign .nat [] "start_n" (.ref .nat [] "start_n"),
   .assign .bool [BD,BN] "k_mask" (keyMask BD BN D),
   .assign .real [BD,BN] "k" (.load .real (.ptr (.ref .ptr [BD,BN] (keyName second)))
     (.maskOther (.ref .bool [BD,BN] "k_mask") (.broadcast (.const 0.0) [BD,BN]))),
   .assign .real [] "k_scale" (.load .real (.ptr (.ref .ptr [] (scaleName second))) .none)]

def guard (BM BN : Nat) (second diagonal : Bool) : List Stmt :=
  if diagonal then
    [.assign .bool [BM,BN] (maskName second) (VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions.maskExpr BM BN),
     .assign .real [BM,BN] "qk" (.where
       (.boolAnd (.consR (.consSame .nil)) (.ref .bool [BM,BN] (maskName second))
         (tail BN))
       (.ref .real [BM,BN] "qk") (.broadcast .negInf [BM,BN]))]
  else
    [.assign .real [BM,BN] "qk" (.where (.remap [BM,BN] (Broadcast.leftIndex (.consL (.consSame .nil))) (tail BN))
      (.ref .real [BM,BN] "qk") (.broadcast .negInf [BM,BN]))]

def valueLoad (BN BD D : Nat) (second : Bool) : Stmt :=
  .assign .real [BN,BD] "v" (.load .real (.ptr (.ref .ptr [BN,BD] (valueName second)))
    (.maskOther (valueMask BN BD D) (.broadcast (.const 0.0) [BN,BD])))

def advance (BN BD : Nat) (second : Bool) : List Stmt :=
  [.assign .ptr [BD,BN] (keyName second) (.ptrAdd .scalarR (.ref .ptr [BD,BN] (keyName second))
     (.mul .nat .nil (.constNat BN) (.ref .nat [] "stride_kn64"))),
   .assign .ptr [] (scaleName second) (.ptrAdd .nil (.ref .ptr [] (scaleName second)) (.constNat 1)),
   .assign .ptr [BN,BD] (valueName second) (.ptrAdd .scalarR (.ref .ptr [BN,BD] (valueName second))
     (.mul .nat .nil (.constNat BN) (.ref .nat [] "stride_vk64")))]

def suffix (BM BN BD D : Nat) (second : Bool) : List Stmt :=
  [VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.probability BM BN] ++ VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.rescale BM BN BD ++
    [valueLoad BN BD D second] ++ VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.accumulate BM BN BD ++ advance BN BD second

def flatIteration (BM BN BD D : Nat) (second diagonal : Bool) : List Stmt :=
  loadPrefix BD BN D second ++ [VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.score BM BN BD] ++ guard BM BN second diagonal ++
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.center BM BN ++ suffix BM BN BD D second

def sourceIteration (BM BN BD D : Nat) (second firstDiagonal : Bool) : List Stmt :=
  if second then flatIteration BM BN BD D Bool.true Bool.true else
    loadPrefix BD BN D Bool.false ++ [VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.score BM BN BD,
      .ifThenElse (.constBool firstDiagonal)
        (guard BM BN Bool.false Bool.true ++ VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.center BM BN)
        (guard BM BN Bool.false Bool.false ++ VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.center BM BN)] ++
      suffix BM BN BD D Bool.false

/-- The rest of the fragments are selected from the actual shared source. -/
def coordinates (BM BN BD : Nat) (p : Parameters) := (code BM BN BD p Bool.false).take 13
def pointers (BM BN BD : Nat) (p : Parameters) := ((code BM BN BD p Bool.false).drop 13).take 9
def initialization (BM BN BD : Nat) (p : Parameters) := ((code BM BN BD p Bool.false).drop 22).take 5
def firstSetup (BM BN BD : Nat) (p : Parameters) := ((code BM BN BD p Bool.false).drop 27).take 6
def secondSetup (BM BN BD : Nat) (p : Parameters) := ((code BM BN BD p Bool.false).drop 34).take 6

def loop (BM BN BD D : Nat) (second firstDiagonal : Bool) : Stmt :=
  .forRangeDyn "start_n" (.ref .nat [] (if second then "second_lo" else "first_lo"))
    (.ref .nat [] (if second then "second_hi" else "first_hi")) (.constNat BN)
    (sourceIteration BM BN BD D second firstDiagonal)

def epilogue (BM BD D : Nat) (r : Bool) : List Stmt :=
  [VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource.normalization BM BD r,
   .store .real [BM,BD] (.ptr (.ref .ptr [BM,BD] "O_block_ptr")) (.ref .real [BM,BD] "acc")
     (.mask (queryMask BM BD D))]

/-- Every statement, both helper loops, and the complete store are included. -/
theorem decomposition (BM BN BD : Nat) (p : Parameters) (r : Bool) :
    code BM BN BD p r = coordinates BM BN BD p ++ pointers BM BN BD p ++ initialization BM BN BD p ++
      firstSetup BM BN BD p ++ [loop BM BN BD p.headDim Bool.false (decide (p.stage=2))] ++
      secondSetup BM BN BD p ++ [loop BM BN BD p.headDim Bool.true (decide (p.stage=2))] ++
      epilogue BM BD p.headDim r := by
  cases r
  all_goals
    simp only [coordinates, pointers, initialization, firstSetup, secondSetup, code, source,
      TypedSource.repaired, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [loop, sourceIteration, flatIteration, loadPrefix, guard, tail, keyMask, valueMask,
      queryMask, valueLoad, advance, suffix, epilogue, keyName, scaleName, valueName, maskName,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.keyName,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.scaleName,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.valueName,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation.maskName,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.score,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.scoreExpr,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.center,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.centerExpr,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.probability,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.rescale,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody.accumulate,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource.normalization,
      VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions.maskExpr]

#guard_msgs (drop info) in
#auditModuleAxioms
end Verified
end VeriTile.Bench.Repairs.AttnFwdTriton
