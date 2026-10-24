import bench.repairs.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHProgram

/-! The repaired recurrence reuses every original arithmetic/load/store
statement and changes only pointer construction plus the launch decoder. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000

abbrev plus (a b : Op .nat []) := Op.add .nat .nil a b
abbrev mul (a b : Op .nat []) := Op.mul .nat .nil a b

def local2 (r : RegionName) (base : Op .nat []) (rows cols sr sc : Nat) : Op .blockPtr [rows,cols] :=
  .makeBlockPtrDyn r base [rows,cols] [rows,cols] [sr,sc] [0,0]
def local1 (r : RegionName) (base : Op .nat []) (size stride : Nat) : Op .blockPtr [size] :=
  .makeBlockPtrDyn r base [size] [size] [stride] [0]

def keyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BT] :=
  local2 r (plus (plus (times "i_bh" p.keyHead)
    (mul (times "i_k" p.BK) (.constNat p.keyFeature)))
    (mul (times "i_t" p.BT) (.constNat p.keyTime))) p.BK p.BT p.keyFeature p.keyTime

def valuePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BT,p.BV] :=
  local2 r (plus (plus (times "i_bh" p.valueHead)
    (mul (times "i_t" p.BT) (.constNat p.valueTime)))
    (mul (times "i_v" p.BV) (.constNat p.valueFeature))) p.BT p.BV p.valueTime p.valueFeature

def statePtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  local2 r (plus (plus (stateBase p) (mul (times "i_k" p.BK) (.constNat p.V)))
    (times "i_v" p.BV)) p.BK p.BV p.V 1

def historyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  local2 r (plus (plus (plus (times "i_bh" p.historyHead)
    (mul (times "i_t" p.K) (.constNat p.V)))
    (mul (times "i_k" p.BK) (.constNat p.historyKey)))
    (mul (times "i_v" p.BV) (.constNat p.historyValue))) p.BK p.BV p.historyKey p.historyValue

def endpointPtr (p : Layout) (r : RegionName) (head time stride block : Nat)
    (index : RegName) : Op .blockPtr [block] :=
  local1 r (plus (plus (times "i_bh" head)
    (mul (.sub .nat .nil (plus (times "i_t" p.BT) (.constNat p.BT)) (.constNat 1)) (.constNat time)))
    (mul (times index block) (.constNat stride))) block stride

def gateKey (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_g" (keyPtr p r.gate),
   .assign .blockPtr [p.BK] "p_gn" (endpointPtr p r.gate p.keyHead p.keyTime p.keyFeature p.BK "i_k")] ++
    (RealHProgram.gateKey p r).drop 2

def gateValue (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BV] "p_g" (valuePtr p r.gate),
   .assign .blockPtr [p.BV] "p_gn" (endpointPtr p r.gate p.valueHead p.valueTime p.valueFeature p.BV "i_v")] ++
    (RealHProgram.gateValue p r).drop 2

def prelude (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_k" (keyPtr p r.key),
   .assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BK,p.BV] "p_h" (historyPtr p r.history)] ++
     (RealHProgram.iterationPrelude p r).drop 3

def iteration (p : Layout) (r : Regions) (gateK swapped : Bool) : List Stmt :=
  prelude p r ++ [.ifThenElse (.constBool gateK)
    (gateKey p r ++ [updateStmt p swapped]) (gateValue p r ++ [updateStmt p swapped])]

def initialization (p : Layout) (r : Regions) (base : Nat) (hasSeed : Bool) : List Stmt :=
  [.assign .nat [] "linear" (plus (.programId 0) (.constNat base)),
   .assign .nat [] "i_v" (.mod .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV))),
   .assign .nat [] "i_k" (.mod .nat .nil
     (.div .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV))) (.constNat (p.K/p.BK))),
   .assign .nat [] "i_bh" (.div .nat .nil (.ref .nat [] "linear") (.constNat ((p.V/p.BV)*(p.K/p.BK)))),
   .assign .real [p.BK,p.BV] "b_h" (.full [p.BK,p.BV] (.const 0)),
   .ifThen (.constBool hasSeed)
     [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.seed),
      .assign .real [p.BK,p.BV] "b_h"
        (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h"))]]

def finalization (p : Layout) (r : Regions) (storeFinal : Bool) : List Stmt :=
  [.ifThen (.constBool storeFinal)
    [.assign .blockPtr [p.BK,p.BV] "p_h" (statePtr p r.final),storeCarry p]]

def body (p : Layout) (r : Regions) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) : List Stmt :=
  initialization p r base hasSeed ++
    [.forRange "i_t" 0 p.NT 1 (iteration p r gateK swapped)] ++ finalization p r storeFinal

def source (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) :=
  (if swapped then NativeSource.hSumLeft else NativeSource.hOriginal) dot gate history seed final
    r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT base
    gateK hasSeed storeFinal

/-- Exact projection check for the repaired typed source, independently of
all runtime correctness assumptions. Precision is erased only here. -/
theorem source_body (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) :
    (source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.body =
      body p r base gateK hasSeed storeFinal swapped := by
  cases swapped <;> cases dot <;> cases gate <;> cases history
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, NativeSource.hOriginal,
    NativeSource.hSumLeft, NativeSource.hSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases seed <;> cases final <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedProgram
