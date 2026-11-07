import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_Prelude
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.PreludeProgram
open VeriTile.Triton Formula Program Prelude
set_option maxHeartbeats 1500000
set_option maxRecDepth 2000
set_option pp.deepTerms false
set_option linter.unusedSimpArgs false

def n (name : RegName) : Op .nat [] := .ref .nat [] name
def z (name : RegName) : Op .int [] := .ref .int [] name
def loadMeta (r : RegionName) : Op .int [] := .load .int (.region r (z "cur_batch").castIntToNat) .none

def launchCode (p : Layout) : List Stmt :=
  [.assign .int [] "cur_batch" (.castNatToInt (.programId 0)),
   .assign .int [] "cur_head" (.castNatToInt (.programId 1)),
   .assign .int [] "start_m" (.castNatToInt (.programId 2)),
   .assign .int [] "cur_kv_head" (.truncDiv .nil (z "cur_head") (.constInt p.kvGroup))]

def metadataCode (r : Regions) : List Stmt :=
  [.assign .int [] "cur_batch_seq_len" (loadMeta r.sequence),
   .assign .int [] "cur_batch_in_all_start_index" (loadMeta r.start)]

def rows (p : Layout) : Op .int [p.BM] := .ref .int [p.BM] "offs_m"
def dimensions (p : Layout) : Op .int [p.D] := .ref .int [p.D] "offs_d"
def rowView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (rows p)
def dimView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (dimensions p)
def offset (p : Layout) : Op .int [p.BM,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR
      (.mul .int .scalarR (.add .int .scalarL (z "cur_batch_in_all_start_index")
        (rowView p)) (Op.constInt p.qRow))
      (Op.mul .int .nil (z "cur_head") (.constInt p.qHead)))
    (Op.mul .int .scalarR (dimView p) (.constInt p.qDim))

def lanes (p : Layout) : Op .int [p.BN] := .ref .int [p.BN] "offs_n"
def keyView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (lanes p)
def keyChannel (p : Layout) := Op.expandDim ⟨1,by simp⟩ (dimensions p)
def valueView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (lanes p)

def keyOffset (p : Layout) : Op .int [p.D,p.BN] :=
  .add .int Broadcast.nil.consR.consL
    (.add .int .scalarR (.mul .int .scalarR (keyView p) (.constInt p.kRow))
      (.mul .int .nil (z "cur_kv_head") (.constInt p.kHead)))
    (.mul .int .scalarR (keyChannel p) (.constInt p.kDim))
def valueOffset (p : Layout) : Op .int [p.BN,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR (.mul .int .scalarR (valueView p) (.constInt p.vRow))
      (.mul .int .nil (z "cur_kv_head") (.constInt p.vHead)))
    (.mul .int .scalarR (dimView p) (.constInt p.vDim))

def offsetCode (p : Layout) : List Stmt :=
  [.assign .int [] "block_start_loc" (Op.mul .int .nil (.constInt p.BM) (z "start_m")),
   .assign .int [p.BN] "offs_n" (.castNatToInt (.arange p.BN)),
   .assign .int [p.D] "offs_d" (.castNatToInt (.arange p.D)),
   .assign .int [p.BM] "offs_m"
      (.add .int .scalarL (.mul .int .nil (z "start_m") (.constInt p.BM)) (.castNatToInt (.arange p.BM))),
   .assign .int [p.BM,p.D] "off_q" (offset p),
   .assign .int [p.D,p.BN] "off_k" (keyOffset p),
   .assign .int [p.BN,p.D] "off_v" (valueOffset p)]

def queryMask (p : Layout) : Op .bool [p.BM,p.D] :=
  .remap [p.BM,p.D] (Broadcast.leftIndex Broadcast.nil.consL.consSame)
    (.lt .int .scalarR (rowView p) (z "cur_batch_seq_len"))
def loadQuery (p : Layout) (r : Regions) : Op .real [p.BM,p.D] :=
  .load .real (.region r.query (Op.ref .int [p.BM,p.D] "off_q").castIntToNat)
    (.maskOther (queryMask p) (.broadcast (.const (0.0 : ℝ)) [p.BM,p.D]))

def queryCode (p : Layout) (r : Regions) : List Stmt :=
  [.assign .real [p.BM,p.D] "q" (loadQuery p r),
   .assign .ptr [p.D,p.BN] "k_ptrs" (.ptrAdd .scalarL (.ptrBase r.key) (Op.ref .int [p.D,p.BN] "off_k").castIntToNat),
   .assign .ptr [p.BN,p.D] "v_ptrs" (.ptrAdd .scalarL (.ptrBase r.value) (Op.ref .int [p.BN,p.D] "off_v").castIntToNat),
   .assign .real [p.BM] "m_i" (.add .real .scalarR (.full [p.BM] (.const 0)) .negInf),
   .assign .real [p.BM] "l_i" (.full [p.BM] (.const 0)),
   .assign .real [p.BM,p.D] "acc" (.full [p.BM,p.D] (.const 0)),
   .assign .int [] "block_mask" (.where
     (.lt .int .nil (z "block_start_loc") (z "cur_batch_seq_len")) (.castNatToInt (.constNat 1)) (.castNatToInt (.constNat 0)))]

theorem erase_view (ax : Fin (shape.length+1)) (e : Op dtype shape) :
    (Op.expandDim ax e).eraseDType = Op.expandDim ax e.eraseDType := by
  rw [Op.eraseDType.eq_def]

theorem source (p : Layout) (r : Regions) :
    prelude p r = launchCode p ++ metadataCode r ++ offsetCode p ++ queryCode p r := by
  rw [prefix_source]
  simp only [prefixSource,parts,mathematical,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,
    VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,
    VeriTile.Bench.Repairs.ContextAttnMistralKernel.originalKernel,
    Source.intRange?,String.reduceEq,Bool.false_eq_true,if_false,if_true,Option.map_some,Option.map_none,Option.getD,Except.toOption,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,List.map_cons,List.map_nil,Region.cast_cast,
    launchCode,metadataCode,offsetCode,queryCode,n,z,loadMeta,rows,dimensions,rowView,dimView,
    offset,queryMask,loadQuery,keyOffset,valueOffset,lanes,keyView,keyChannel,valueView,List.cons_append,List.nil_append]
  try simp only [erase_view,Op.eraseDType_ref,VeriTile.Triton.eraseDType,Region.cast_id]
  try simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.PreludeProgram
