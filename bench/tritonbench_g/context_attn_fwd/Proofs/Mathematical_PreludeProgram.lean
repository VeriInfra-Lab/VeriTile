import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_Prelude
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeProgram
open VeriTile.Triton Formula Program Prelude
set_option maxHeartbeats 1500000
set_option maxRecDepth 2000
set_option pp.deepTerms false
set_option linter.unusedSimpArgs false

def n (name : RegName) : Op .nat [] := .ref .nat [] name
def z (name : RegName) : Op .int [] := .ref .int [] name
def loadMeta (r : RegionName) : Op .int [] := .load .int (.region r (z "cur_batch").castIntToNat) .none

def launchCode (p : Layout) : List Stmt :=
  [.assign .int [] "start_m" (.castNatToInt (.programId 0)),
   .assign .int [] "cur_bh" (.castNatToInt (.programId 1)),
   .assign .int [] "cur_batch" (.truncDiv .nil (z "cur_bh") (.constInt p.heads)),
   .assign .int [] "cur_head" (.truncMod .nil (z "cur_bh") (.constInt p.heads)),
   .assign .int [] "cur_kv_head" (.truncDiv .nil (z "cur_head") (.constInt p.kvGroup))]

def metadataCode (r : Regions) : List Stmt :=
  [.assign .int [] "prompt_cache_len" (loadMeta r.prompt),
   .assign .int [] "cur_batch_in_all_start_index" (loadMeta r.start),
   .assign .int [] "cur_batch_seq_len" (.sub .int .nil (loadMeta r.sequence) (z "prompt_cache_len"))]

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

def offsetCode (p : Layout) : List Stmt :=
  [.assign .int [] "block_start_loc" (Op.mul .int .nil (.constInt p.BM) (z "start_m")),
   .assign .int [p.BN] "offs_n" (.castNatToInt (.arange p.BN)),
   .assign .int [p.D] "offs_d" (.castNatToInt (.arange p.D)),
   .assign .int [p.BM] "offs_m"
      (.add .int .scalarL (z "block_start_loc") (.castNatToInt (.arange p.BM))),
   .assign .int [p.BM,p.D] "off_q" (offset p)]

def queryMask (p : Layout) : Op .bool [p.BM,p.D] :=
  .remap [p.BM,p.D] (Broadcast.leftIndex Broadcast.nil.consL.consSame)
    (.lt .int .scalarR (rowView p) (z "cur_batch_seq_len"))
def loadQuery (p : Layout) (r : Regions) : Op .real [p.BM,p.D] :=
  .load .real (.region r.query (Op.ref .int [p.BM,p.D] "off_q").castIntToNat)
    (.maskOther (queryMask p) (.broadcast (.const (0.0 : ℝ)) [p.BM,p.D]))

def endCandidate (p : Layout) : Op .int [] :=
  .add .int .nil
    (Op.add .int .nil (z "block_start_loc") (.castNatToInt (.constNat p.BM)))
    (z "prompt_cache_len")
def fullLength : Op .int [] := .add .int .nil (z "cur_batch_seq_len") (z "prompt_cache_len")
def ending (p : Layout) : Op .int [] :=
  .where (.lt .int .nil (endCandidate p) fullLength) (endCandidate p) fullLength

def queryCode (p : Layout) (r : Regions) : List Stmt :=
  [.assign .real [p.BM,p.D] "q" (loadQuery p r),
   .assign .real [p.BM] "m_i" (.add .real .scalarR (.full [p.BM] (.const 0)) .negInf),
   .assign .real [p.BM] "l_i" (.full [p.BM] (.const 0)),
   .assign .real [p.BM,p.D] "acc" (.full [p.BM,p.D] (.const 0)),
   .assign .int [] "block_mask" (.where
     (.lt .int .nil (z "block_start_loc") (z "cur_batch_seq_len")) (.castNatToInt (.constNat 1)) (.castNatToInt (.constNat 0))),
   .assign .int [] "block_end_loc" (ending p)]

theorem erase_view (ax : Fin (shape.length+1)) (e : Op dtype shape) :
    (Op.expandDim ax e).eraseDType = Op.expandDim ax e.eraseDType := by
  rw [Op.eraseDType.eq_def]

theorem source (p : Layout) (r : Regions) :
    prelude p r = launchCode p ++ metadataCode r ++ offsetCode p ++ queryCode p r := by
  rw [prefix_source]
  simp only [prefixSource,parts,mathematical,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,
    VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original,
    VeriTile.Bench.Repairs.ContextAttnFwd.originalKernel,Source.intRange?,String.reduceEq,
    Bool.false_eq_true,if_false,if_true,Option.map_some,Option.map_none,Option.getD,Except.toOption,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,List.map_cons,List.map_nil,Region.cast_cast,
    launchCode,metadataCode,offsetCode,queryCode,n,z,loadMeta,rows,dimensions,rowView,dimView,
    offset,queryMask,loadQuery,endCandidate,fullLength,ending,List.cons_append,List.nil_append]
  try simp only [erase_view,Op.eraseDType_ref,VeriTile.Triton.eraseDType,Region.cast_id]
  try simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.PreludeProgram
