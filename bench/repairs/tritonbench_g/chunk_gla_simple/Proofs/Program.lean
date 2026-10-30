import bench.repairs.tritonbench_g.chunk_gla_simple.Kernels
import bench.tritonbench_g.chunk_gla_simple.Proofs.RealExecution

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Program
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def times (name : RegName) (n : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] name) (.constNat n)
def indices (name : RegName) (n : Nat) : Op .nat [n] :=
  .add .nat .scalarL (times name n) (.arange n)
def col (name : RegName) (n : Nat) : Op .nat [n,1] :=
  .expandDim ⟨1,by simp⟩ (.ref .nat [n] name)
def row (name : RegName) (n : Nat) : Op .nat [1,n] :=
  .expandDim ⟨0,by simp⟩ (.ref .nat [n] name)
def scaleCol (name : RegName) (n stride : Nat) : Op .nat [n,1] :=
  .mul .nat .scalarR (col name n) (.constNat stride)
def scaleRow (name : RegName) (n stride : Nat) : Op .nat [1,n] :=
  .mul .nat .scalarR (row name n) (.constNat stride)
def ptr2 (r : RegionName) (base : Op .nat []) (x : Op .nat [m,1]) (y : Op .nat [1,n]) : Op .ptr [m,n] :=
  .ptrAdd .scalarL (.ptrBase r)
    (.add .nat (.consR (.consL .nil)) (.add .nat .scalarL base x) y)
def mask2 (x y : RegName) (m n rows cols : Nat) : Op .bool [m,n] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (col x m) (.constNat rows))
    (.lt .nat .scalarR (row y n) (.constNat cols))
def load2 (ptr x y : RegName) (m n rows cols : Nat) : Op .real [m,n] :=
  .load .real (.ptr (.ref .ptr [m,n] ptr))
    (.maskOther (mask2 x y m n rows cols) (.broadcast (.const 0) [m,n]))
def historyBase (p : Layout) : Op .nat [] :=
  .add .nat .nil (times "i_bh" p.historyHead)
    (.mul .nat .nil (times "i_t" p.K) (.constNat p.V))
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
    (.add .nat .scalarL (times "i_bh" p.T) (.ref .nat [p.BT] "rows"))
def gateLoad (p : Layout) : Op .real [p.BT] :=
  .load .real (.ptr (.ref .ptr [p.BT] "p_g"))
    (.maskOther (.lt .nat .scalarR (.ref .nat [p.BT] "rows") (.constNat p.T))
      (.broadcast (.const 0) [p.BT]))
def addresses (p : Layout) : List Stmt :=
  [.assign .nat [p.BT] "rows" (indices "i_t" p.BT),
   .assign .nat [p.BV] "cols" (indices "i_v" p.BV)]
def setup (p : Layout) : List Stmt := Real.setup p ++ addresses p

def iteration (p : Layout) (r : Regions) : List Stmt :=
  [.assign .nat [p.BK] "keys" (indices "i_k" p.BK),
   .assign .ptr [p.BT,p.BK] "p_q" (queryPtr p r),
   .assign .ptr [p.BK,p.BT] "p_k" (keyPtr p r),
   .assign .ptr [p.BK,p.BV] "p_h" (historyPtr p r),
   .assign .real [p.BT,p.BK] "b_q" (load2 "p_q" "rows" "keys" p.BT p.BK p.T p.K),
   .assign .real [p.BK,p.BT] "b_k" (load2 "p_k" "keys" "rows" p.BK p.BT p.K p.T),
   .assign .real [p.BK,p.BV] "b_h" (load2 "p_h" "keys" "cols" p.BK p.BV p.K p.V)] ++
  (Real.iteration p r).drop 6

def loop (p : Layout) (r : Regions) : Stmt :=
  .forRangeDyn "i_k" (.constNat 0) (Real.tripCount p) (.constNat 1) (iteration p r)

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
  ChunkGlaSimple.source .fp32 .fp32 .fp32 Bool.false .fp32
    r.query r.key r.value r.history r.gate r.output
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale
    p.T p.K p.V p.BT p.BK p.BV

/-- Exact binding of the FP32 repaired typed source, with casts erased only for
its independent Real interpretation. Unchanged arithmetic statements are
literally reused from the original program, rather than reproved here. -/
theorem source_body (p : Layout) (r : Regions) (scale : ℝ) :
    (typed p r scale).eraseDType.toAlgKernel.body = body p r scale := by
  simp only [typed, ChunkGlaSimple.source, Bool.false_eq_true, ↓reduceIte,
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
