import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeProjection
import VeriTile.Triton.Float.IntegerErasure
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Semantics.Step
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def algorithm (p : Parameters) (s o : RegionName) : Kernel :=
  (projection p .real .real s o).val.eraseDType

local macro "reduce_raw_source" h:term : tactic => `(tactic|
  (dsimp only [algorithm,projection,Kernel.eraseDType]
   simp only [($h),Bool.false_eq_true,↓reduceDIte,↓reduceIte]
   simp only [ScalarArgument.integer,ScalarArgument.statement,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType_truncDiv,Op.eraseDType_truncMod,Op.eraseDType_castFloat,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]))

theorem source_connection (p : Parameters) (s o : RegionName) :
    (kernel p .real .real s o).eraseDType.toAlgKernel=algorithm p s o :=
  erase_connection p .real .real s o


def prelude (p : Parameters) (s o : RegionName) : List Stmt :=
  (algorithm p s o).body.take 4

def guardStatement (p : Parameters) (s o : RegionName) : Stmt :=
  ((algorithm p s o).body[4]?).getD (.ifThen (.constBool Bool.false) [])

def activeBody (p : Parameters) (s o : RegionName) : List Stmt :=
  match guardStatement p s o with
  | .ifThen _ body => body
  | _ => []

theorem decomposition (p : Parameters) (s o : RegionName) :
    (algorithm p s o).body=prelude p s o ++ [guardStatement p s o] := by
  by_cases h:p.swapped
  all_goals
    unfold prelude guardStatement
    reduce_raw_source h
    rfl

theorem active_length (p : Parameters) (s o : RegionName) :
    (activeBody p s o).length=11 := by
  by_cases h:p.swapped
  all_goals
    unfold activeBody guardStatement
    reduce_raw_source h
    rfl

theorem prelude_ast (p : Parameters) (s o : RegionName) :
    prelude p s o=
      [.assign .int [] "s_s_h" (.constInt p.headStride),
       .assign .int [] "s_s_t" (.constInt p.timeStride),
       .assign .int [] "s_s_d" (.constInt p.featureStride),
       .assign .int [] "linear" (.add .int .nil (.castNatToInt (.programId 0))
         (.castNatToInt (.constNat p.base)))] := by
  by_cases h:p.swapped
  all_goals
    unfold prelude
    reduce_raw_source h
    rfl

def preludeExecution (p : Parameters) (s o : RegionName) (initial : BlockState) :
    {final : BlockState // stepStmts (prelude p s o) initial=some final} := by
  refine ⟨?_,?_⟩
  swap
  simp only [prelude_ast,stepStmts,stepStmt,VeriTile.Triton.evalOp,Tile.bop,Tile.scalar,
    Option.bind_some,Option.bind_eq_bind,Option.map_some,Option.pure_def]
  rfl

theorem prelude_memory (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (preludeExecution p s o initial).val.mem=initial.mem := by
  simp [preludeExecution]
  rfl

theorem prelude_linear (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (preludeExecution p s o initial).val.regs .int [] "linear"=
      some (Tile.scalar ((initial.pids 0+p.base : Nat) : Int)) := by
  simp [preludeExecution,Int.natCast_add]
  rfl

#axiomsClean source_connection
#axiomsClean decomposition
#axiomsClean active_length
#axiomsClean preludeExecution
#axiomsClean prelude_linear
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
