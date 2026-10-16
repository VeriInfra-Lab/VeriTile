import bench.tritonbench_g.layer_norm_triton.ConcurrentCorrect

namespace VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
open VeriTile.Triton
open FP.Equational (ReductionPlan)

inductive FSlot where | meanSum | mean | varSum | variance | rstd | eps
  deriving DecidableEq, Repr
inductive FExpr where
  | slot (name : FSlot)
  | x | w | bias | zero | one | denominator
  | add (a b : FExpr) | sub (a b : FExpr) | mul (a b : FExpr) | div (a b : FExpr)
  | sqrt (a : FExpr)
  | promoteW (a : FExpr) | promoteBias (a : FExpr)
  | mask (a : FExpr)
  deriving DecidableEq, Repr

def evalF (b : Backend V) (c : Config) (mem : Memory V) (undef : Undef V)
    (p : Nat) (env : FSlot → V) (off : I32) (lane : Nat) : FExpr → V
  | .slot name => env name
  | .x => xRead b c mem p off lane
  | .w => if live c off lane then mem .w (colAt off lane).toInt else undef .w (colAt off lane).toInt
  | .bias => if live c off lane then mem .bias (colAt off lane).toInt else undef .bias (colAt off lane).toInt
  | .zero => b.zero
  | .one => b.one
  | .denominator => b.fromI32 c.n
  | .add a d => b.add (evalF b c mem undef p env off lane a) (evalF b c mem undef p env off lane d)
  | .sub a d => b.sub (evalF b c mem undef p env off lane a) (evalF b c mem undef p env off lane d)
  | .mul a d => b.mul (evalF b c mem undef p env off lane a) (evalF b c mem undef p env off lane d)
  | .div a d => b.div (evalF b c mem undef p env off lane a) (evalF b c mem undef p env off lane d)
  | .sqrt a => b.sqrt (evalF b c mem undef p env off lane a)
  | .promoteW a => b.cast c.dtype.w .fp32 (evalF b c mem undef p env off lane a)
  | .promoteBias a => b.cast c.dtype.bias .fp32 (evalF b c mem undef p env off lane a)
  | .mask a => if live c off lane then evalF b c mem undef p env off lane a else b.zero

structure ForwardPlan where
  meanInput : FExpr
  meanResult : FExpr
  varInput : FExpr
  varResult : FExpr
  rstdResult : FExpr
  yResult : FExpr
  deriving DecidableEq, Repr

/-- Plans carry source expression trees, not opaque callbacks/results. The
source compiler materializes only the source type promotion at W/B uses. -/
def executeForward (plan : ForwardPlan) (b : Backend V) (c : Config)
    (reduction : ReductionPlan c.block) (mem : Memory V) (undef : Undef V)
    (eps : V) (p : Nat) (offsets : List I32) : ForwardValues V :=
  let env0 := put (fun _ => b.zero) .eps eps
  let meanReg := streamRun b (fun off lane => evalF b c mem undef p env0 off lane plan.meanInput) offsets (fun _ => b.zero)
  let env1 := put env0 .meanSum (sum b reduction meanReg)
  let mean := evalF b c mem undef p env1 0 0 plan.meanResult
  let env2 := put env1 .mean mean
  let varReg := streamRun b (fun off lane => evalF b c mem undef p env2 off lane plan.varInput) offsets (fun _ => b.zero)
  let env3 := put env2 .varSum (sum b reduction varReg)
  let variance := evalF b c mem undef p env3 0 0 plan.varResult
  let env4 := put env3 .variance variance
  let inv := evalF b c mem undef p env4 0 0 plan.rstdResult
  let env5 := put env4 .rstd inv
  ⟨mean, inv, fun off lane => b.cast .fp32 c.dtype.y (evalF b c mem undef p env5 off lane plan.yResult)⟩

structure LockPlan where
  expected : I32
  desired : I32
  retry : I32
  initialCount : I32
  markCount : I32
  releaseValue : I32
  deriving DecidableEq, Repr

def decodePlan (plan : LockPlan) (c : Config) (p : Nat) (s : Control) : Event × Phase :=
  if p ≥ c.grid then (.silent, s.pc p) else
  match s.pc p with
  | .poll => (.poll (lockAddr c p), if s.words (lockAddr c p) == plan.retry then .poll else .count)
  | .count => (.readCount (countAddr c p), if s.words (countAddr c p) == plan.initialCount then .setCount else .loadDW 0)
  | _ => decode c p s

def integerPlan (plan : LockPlan) (e : Event) (words : Int → I32) : Int → I32 :=
  match e with
  | .poll a => put words a (if words a == plan.expected then plan.desired else words a)
  | .setCount a => put words a plan.markCount
  | .release a => put words a plan.releaseValue
  | _ => words

def controlStepPlan (plan : LockPlan) (c : Config) (p : Nat) (s : Control) : Event × Control :=
  let (e, next) := decodePlan plan c p s
  (e, ⟨integerPlan plan e s.words, put s.pc p next⟩)

def sourceStep (plan : LockPlan) (b : Backend V) (c : Config)
    (rows : Nat → RowValues V) (undef : Undef V) (p : Nat) (s : State V) : Event × State V :=
  let (e, next) := controlStepPlan plan c p s.control
  (e, ⟨next, execute b c rows undef e s.values⟩)

def sourceRun (plan : LockPlan) (b : Backend V) (c : Config)
    (rows : Nat → RowValues V) (undef : Undef V) : List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, next) := sourceStep plan b c rows undef p s
    let (events, finalState) := sourceRun plan b c rows undef ps next
    (e :: events, finalState)

/-- Exact source range schema, masks and access roles are typed nodes; pure
address prelude is validated by the external typed frontend and preserved in
its statement ledger. No opaque floating result is an instruction. -/
inductive FStmt where
  | setup
  | initializeMean
  | meanLoop (read : FExpr)
  | meanResult (expr : FExpr)
  | initializeVariance
  | varianceLoop (expr : FExpr)
  | varianceResult (expr : FExpr)
  | inverseResult (expr : FExpr)
  | storeMean | storeRstd
  | outputLoop (expr : FExpr)
  deriving DecidableEq, Repr

inductive BStmt where
  | setup
  | assign (name : Nat) (expr : Expr)
  | storeDX (expr : Expr)
  | whileCAS (expected desired retry : I32)
  | loadCount
  | countBranch (compare mark : I32)
  | storeDW | storeDB
  | release (value : I32)
  deriving DecidableEq, Repr

inductive DStmt where
  | setup | zeroDW | zeroDB | streamDWDB | reduceDW | reduceDB | storeDW | storeDB
  deriving DecidableEq, Repr

structure TypedSource where
  forward : List FStmt
  backward : List BStmt
  finalReduce : List DStmt
  deriving DecidableEq, Repr

structure Compiled where
  forward : ForwardPlan
  backwardProgram : List Assignment
  lock : LockPlan
  deriving DecidableEq, Repr

/-- Accept the COMPLETE three-kernel structure, preserving each graph expression
and each atomic/control literal. Any missing/extra/reordered stage is rejected.
The external typed frontend validates all syntax inside the compound nodes. -/
def compile (s : TypedSource) : Option Compiled := do
  let [.setup, .initializeMean, .meanLoop mi, .meanResult mr, .initializeVariance,
    .varianceLoop vi, .varianceResult vr, .inverseResult rr, .storeMean, .storeRstd, .outputLoop y] := s.forward | none
  let [.setup,
    .assign 0 x, .assign 1 dy, .assign 2 w, .assign 3 mean, .assign 4 rstd,
    .assign 5 xhat, .assign 6 wdy, .assign 5 maskedXhat, .assign 6 maskedWdy,
    .assign 7 c1, .assign 8 c2, .assign 9 dx, .storeDX (.reg 9),
    .assign 10 dw, .assign 11 db, .whileCAS expected desired retry,
    .loadCount, .countBranch count mark, .storeDW, .storeDB, .release releaseValue] := s.backward | none
  let [.setup, .zeroDW, .zeroDB, .streamDWDB, .reduceDW, .reduceDB, .storeDW, .storeDB] := s.finalReduce | none
  return ⟨⟨mi,mr,vi,vr,rr,y⟩,
    [(0,x),(1,dy),(2,w),(3,mean),(4,rstd),(5,xhat),(6,wdy),(5,maskedXhat),(6,maskedWdy),(7,c1),(8,c2),(9,dx),(10,dw),(11,db)],
    ⟨expected,desired,retry,count,mark,releaseValue⟩⟩

def compiledRows (p : Compiled) (b : Backend V) (c : Config) (reduction : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (pid : Nat) : RowValues V :=
  let env := runAssignments b c reduction mem undef pid p.backwardProgram (fun _ _ => b.zero)
  ⟨fun lane => b.cast .fp32 c.dtype.dx (env 9 lane), env 10, env 11⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
