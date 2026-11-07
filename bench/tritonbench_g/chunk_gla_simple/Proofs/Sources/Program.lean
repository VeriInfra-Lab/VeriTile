import bench.tritonbench_g.chunk_gla_simple.Proofs.Sources.Kernels
import bench.tritonbench_g.chunk_gla_simple.Proofs.Sources.Mathematical_RealExecution

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Program
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def times (name : RegName) (n : Nat) : Op .int [] := .mul .int .nil (.ref .int [] name) (.castNatToInt (.constNat n))
def indices (name : RegName) (n : Nat) : Op .int [n] :=
  .add .int .scalarL (times name n) (.castNatToInt (.arange n))
def col (name : RegName) (n : Nat) : Op .int [n,1] :=
  .expandDim ⟨1,by simp⟩ (.ref .int [n] name)
def row (name : RegName) (n : Nat) : Op .int [1,n] :=
  .expandDim ⟨0,by simp⟩ (.ref .int [n] name)
def scaleCol (name : RegName) (n stride : Nat) : Op .int [n,1] :=
  .mul .int .scalarR (col name n) (.castNatToInt (.constNat stride))
def scaleRow (name : RegName) (n stride : Nat) : Op .int [1,n] :=
  .mul .int .scalarR (row name n) (.castNatToInt (.constNat stride))
def ptr2 (r : RegionName) (base : Op .int []) (x : Op .int [m,1]) (y : Op .int [1,n]) : Op .ptr [m,n] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.castIntToNat (.add .int (.consR (.consL .nil)) (.add .int .scalarL base x) y))
def mask2 (x y : RegName) (m n rows cols : Nat) : Op .bool [m,n] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .int .scalarR (col x m) (.castNatToInt (.constNat rows)))
    (.lt .int .scalarR (row y n) (.castNatToInt (.constNat cols)))
def load2 (ptr x y : RegName) (m n rows cols : Nat) : Op .real [m,n] :=
  .load .real (.ptr (.ref .ptr [m,n] ptr))
    (.maskOther (mask2 x y m n rows cols) (.broadcast (.const 0) [m,n]))
def historyBase (p : Layout) : Op .int [] :=
  .add .int .nil (times "i_bh" p.historyHead)
    (.mul .int .nil (times "i_t" p.K) (.castNatToInt (.constNat p.V)))
def queryPtr (p : Layout) (r : Regions) : Op .ptr [p.BT,p.BK] :=
  ptr2 r.query (times "i_bh" p.keyHead) (scaleCol "rows" p.BT p.keyTime) (row "keys" p.BK)
def keyPtr (p : Layout) (r : Regions) : Op .ptr [p.BK,p.BT] :=
  ptr2 r.key (times "i_bh" p.keyHead) (col "keys" p.BK) (scaleRow "rows" p.BT p.keyTime)
def historyPtr (p : Layout) (r : Regions) : Op .ptr [p.BK,p.BV] :=
  ptr2 r.history (historyBase p) (scaleCol "keys" p.BK p.historyKey) (row "cols" p.BV)
def valuePtr (p : Layout) (r : RegionName) : Op .ptr [p.BT,p.BV] :=
  ptr2 r (times "i_bh" p.valueHead) (scaleCol "rows" p.BT p.valueTime) (row "cols" p.BV)
def gatePtr (p : Layout) (r : Regions) : Op .ptr [p.BT] :=
  .ptrAdd .scalarL (.ptrBase r.gate)
    (.castIntToNat (.add .int .scalarL (times "i_bh" p.T) (.ref .int [p.BT] "rows")))
def gateLoad (p : Layout) : Op .real [p.BT] :=
  .load .real (.ptr (.ref .ptr [p.BT] "p_g"))
    (.maskOther (.lt .int .scalarR (.ref .int [p.BT] "rows") (.castNatToInt (.constNat p.T)))
      (.broadcast (.const 0) [p.BT]))
def programIndices (name : RegName) (n : Nat) : Op .int [n] :=
  .add .int .scalarL (.mul .int .nil (.ref .int [] name) (.constInt (Int.ofNat n)))
    (.castNatToInt (.arange n))
def addresses (p : Layout) : List Stmt :=
  [.assign .int [p.BT] "rows" (programIndices "i_t" p.BT),
   .assign .int [p.BV] "cols" (programIndices "i_v" p.BV)]
def setup (p : Layout) : List Stmt :=
  [.assign .int [] "i_v" (.castNatToInt (.programId 0)),
   .assign .int [] "i_t" (.castNatToInt (.programId 1)),
   .assign .int [] "i_bh" (.castNatToInt (.programId 2)),
   .assign .int [] "s_k_h" (.castNatToInt (.constNat p.keyHead)),
   .assign .int [] "s_k_t" (.castNatToInt (.constNat p.keyTime)),
   .assign .int [] "s_v_h" (.castNatToInt (.constNat p.valueHead)),
   .assign .int [] "s_v_t" (.castNatToInt (.constNat p.valueTime)),
   .assign .int [] "s_h_h" (.castNatToInt (.constNat p.historyHead)),
   .assign .int [] "s_h_t" (.castNatToInt (.constNat p.historyKey))] ++
   (Real.setup p).drop 3 ++ addresses p

def iteration (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [p.BK] "keys" (indices "i_k" p.BK),
   .assign .ptr [p.BT,p.BK] "p_q" (queryPtr p r),
   .assign .ptr [p.BK,p.BT] "p_k" (keyPtr p r),
   .assign .ptr [p.BK,p.BV] "p_h" (historyPtr p r),
   .assign .real [p.BT,p.BK] "b_q" (load2 "p_q" "rows" "keys" p.BT p.BK p.T p.K),
   .assign .real [p.BK,p.BT] "b_k" (load2 "p_k" "keys" "rows" p.BK p.BT p.K p.T),
   .assign .real [p.BK,p.BV] "b_h" (load2 "p_h" "keys" "cols" p.BK p.BV p.K p.V)] ++
  (Real.iteration p r).drop 6

def loop (p : Layout) (r : Regions) : Stmt :=
  .forRangeInt "i_k" (.constInt 0) (.castNatToInt (Real.tripCount p)) (.constNat 1) (iteration p r)

def tail (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  [.assign .ptr [p.BT] "p_g" (gatePtr p r),
   .assign .real [p.BT] "b_g" (gateLoad p)] ++
  ((Real.tail p r scale).drop 2).take 3 ++
  [.assign .ptr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .real [p.BT,p.BV] "b_v" (load2 "p_v" "rows" "cols" p.BT p.BV p.T p.V)] ++
  ((Real.tail p r scale).drop 7).take 1 ++
  [.assign .ptr [p.BT,p.BV] "p_o" (valuePtr p r.output)]

def store (p : Layout) : Stmt :=
  .store .real [p.BT,p.BV] (.ptr (.ref .ptr [p.BT,p.BV] "p_o")) (.ref .real [p.BT,p.BV] "b_o")
    (.mask (mask2 "rows" "cols" p.BT p.BV p.T p.V))

def body (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  setup p ++ [loop p r] ++ tail p r scale ++ [store p]

def typed (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  ChunkGlaSimple.source .fp32
    r.query r.key r.value r.history r.gate r.output
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale
    p.T p.K p.V p.BT p.BK p.BV

/-- Exact binding of the FP32 repaired typed source, with casts erased only for
its independent Real interpretation. Unchanged arithmetic statements are
literally reused from the original program, rather than reproved here. -/
theorem source_body (p : Layout) (r : Regions) (scale : ℝ) :
    (typed p r scale).eraseDType.toAlgKernel.body = body p r scale := by
  simp only [typed, ChunkGlaSimple.source,VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.fp32Original,
    VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.originalKernel, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimple.Program
