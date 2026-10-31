import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasure
import VeriTile.Triton.Float.IntegerErasure
import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Semantics.Step
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

local macro "reduce_source" h:term : tactic => `(tactic|
  (dsimp only [algorithm,projection,Kernel.eraseDType]
   simp only [($h),Bool.false_eq_true,↓reduceDIte,↓reduceIte]
   simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,VeriTile.Triton.eraseDType,FloatDType.toTileDType,List.map_cons,List.map_nil]))

theorem source_connection (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (kernel p r dot gate history seed final).eraseDType.toAlgKernel=algorithm p r := by
  rw [erase_connection,erased_algorithm]

def prelude (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  (algorithm p r).body.take 13

def seedStage (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  ((algorithm p r).body.drop 13).take 2

def loopStatement (p : Parameters) (r : RealHProgram.Regions) : Stmt :=
  ((algorithm p r).body[15]?).getD (.ifThen (.constBool Bool.false) [])

def iteration (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  match loopStatement p r with
  | .forRangeInt _ _ _ _ body=>body
  | _=>[]

def finalStage (p : Parameters) (r : RealHProgram.Regions) : List Stmt :=
  (algorithm p r).body.drop 16

theorem decomposition (p : Parameters) (r : RealHProgram.Regions) :
    (algorithm p r).body=prelude p r++seedStage p r++[loopStatement p r]++finalStage p r := by
  unfold prelude seedStage loopStatement finalStage
  by_cases h:p.swapped
  all_goals
    reduce_source h
    rfl

theorem iteration_structure (p : Parameters) (r : RealHProgram.Regions) :
    loopStatement p r=.forRangeInt "i_t" (.castNatToInt (.constNat 0))
      (.castNatToInt (.constNat p.NT)) (.castIntToNat (.castNatToInt (.constNat 1)))
      (iteration p r) := by
  unfold iteration loopStatement
  by_cases h:p.swapped
  all_goals
    reduce_source h
    rfl

theorem prelude_ast (p : Parameters) (r : RealHProgram.Regions) :
    prelude p r=
      [.assign .int [] "s_k_h" (.constInt p.keyHead),
       .assign .int [] "s_k_t" (.constInt p.keyTime),
       .assign .int [] "s_k_d" (.constInt p.keyFeature),
       .assign .int [] "s_v_h" (.constInt p.valueHead),
       .assign .int [] "s_v_t" (.constInt p.valueTime),
       .assign .int [] "s_v_d" (.constInt p.valueFeature),
       .assign .int [] "s_h_h" (.constInt p.historyHead),
       .assign .int [] "s_h_t" (.constInt p.historyKey),
       .assign .int [] "s_h_d" (.constInt p.historyValue),
       .assign .int [] "linear" (.add .int .nil (.castNatToInt (.programId 0))
         (.castNatToInt (.constNat p.base))),
       .assign .int [] "i_v" (.truncMod .nil (.ref .int [] "linear")
         (.castNatToInt (.constNat (p.V/p.BV)))),
       .assign .int [] "i_k" (.truncMod .nil
         (.truncDiv .nil (.ref .int [] "linear") (.castNatToInt (.constNat (p.V/p.BV))))
         (.castNatToInt (.constNat (p.K/p.BK)))),
       .assign .int [] "i_bh" (.truncDiv .nil (.ref .int [] "linear")
         (.castNatToInt (.mul .nat .nil (.constNat (p.V/p.BV)) (.constNat (p.K/p.BK)))))] := by
  unfold prelude
  by_cases h:p.swapped
  all_goals
    reduce_source h
    rfl

def preludeExecution (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    {final : BlockState // stepStmts (prelude p r) initial=some final} := by
  refine ⟨?_,?_⟩
  swap
  rw [prelude_ast]
  simp [-Tile.scalar_eta,stepStmts,stepStmt,VeriTile.Triton.eval_truncDiv,
    VeriTile.Triton.eval_truncMod,VeriTile.Triton.evalOp,Tile.scalar,Tile.bop,
    Option.bind_some,Option.bind_eq_bind,Option.map_some,Option.pure_def,
    BlockState.setReg_same,BlockState.setReg_ne_name]
  rfl

theorem prelude_memory (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.mem=initial.mem := by
  rfl

theorem prelude_pids (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.pids=initial.pids := by
  rfl

theorem prelude_undef (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.undef=initial.undef := by
  rfl

#axiomsClean source_connection
#axiomsClean decomposition
#axiomsClean iteration_structure
#axiomsClean prelude_ast
#axiomsClean preludeExecution
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
