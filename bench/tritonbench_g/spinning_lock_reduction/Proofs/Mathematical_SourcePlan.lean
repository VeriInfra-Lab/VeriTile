import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_ConcurrentCorrect

/-! A source frontend IR, distinct from the executable state machine. Integer
expressions keep signed i32 operators; pointer additions keep separate signed
offsets. Symbolic substitution of pure assignments is performed by the pinned
AST translator. The exact complete typed source tree is generated separately.
No float is translated to Real. The compiler below checks the ENTIRE control
schema, including both while nodes and the two branches, before producing a
plan interpreted by `decodePlan`.
-/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.Frontend

inductive Param where
  | numSms | groupK | bm | bn | m | n | strideM | strideN
  deriving DecidableEq, BEq, Repr
inductive IExpr where
  | literal (value : I32)
  | param (name : Param)
  | pid | next | row | col
  | add (a b : IExpr)
  | mul (a b : IExpr)
  | sdiv (a b : IExpr)
  | srem (a b : IExpr)
  deriving DecidableEq, BEq, Repr
inductive BExpr where
  | eq (a b : IExpr)
  | lt (a b : IExpr)
  | and (a b : BExpr)
  deriving DecidableEq, BEq, Repr
inductive Region where
  | partials | output | locks
  deriving DecidableEq, BEq, Repr
structure Pointer where
  region : Region
  offsets : List IExpr
  deriving DecidableEq, BEq, Repr
inductive Condition where
  | pure (test : BExpr)
  | casNe (pointer : Pointer) (compare replace expected : IExpr)
  deriving DecidableEq, BEq, Repr

/-- The original assignments are retained in the tree even when later uses
carry their fully substituted pure expressions. `range` is the original 1-D
arange; row/column projection occurs at the source [:,None]/[None,:] use. -/
inductive Statement where
  | integer (name : String) (value : IExpr)
  | range (name : String) (size : Param)
  | pointer (name : String) (value : Pointer)
  | mask (name : String) (value : BExpr)
  | zeroFP32 (name : String) (rows cols : Param)
  | loadFP32 (name : String) (pointer : Pointer)
  | addFP32 (target input : String)
  | storeFP32 (pointer : Pointer) (value : String) (mask : Option BExpr)
  | xchg (pointer : Pointer) (value : IExpr)
  | forRange (index : String) (start stop : Nat) (body : List Statement)
  | branch (test : BExpr) (yes no : List Statement)
  | whileLoop (test : Condition) (body : List Statement)
  | pass
  deriving Repr

def evalI (c : Config) (p : Nat) (next : I32) (lane : Nat) : IExpr → I32
  | .literal x => x
  | .param x => match x with
    | .numSms => c.numSms | .groupK => c.groupK
    | .bm => BitVec.ofNat 32 c.bm | .bn => BitVec.ofNat 32 c.bn
    | .m => c.m | .n => c.n | .strideM => c.strideM | .strideN => c.strideN
  | .pid => c.pid p
  | .next => next
  | .row => BitVec.ofNat 32 (lane / c.bn)
  | .col => BitVec.ofNat 32 (lane % c.bn)
  | .add a b => evalI c p next lane a + evalI c p next lane b
  | .mul a b => evalI c p next lane a * evalI c p next lane b
  | .sdiv a b => (evalI c p next lane a).sdiv (evalI c p next lane b)
  | .srem a b => (evalI c p next lane a).srem (evalI c p next lane b)

def evalB (c : Config) (p : Nat) (next : I32) (lane : Nat) : BExpr → Bool
  | .eq a b => evalI c p next lane a == evalI c p next lane b
  | .lt a b => (evalI c p next lane a).slt (evalI c p next lane b)
  | .and a b => evalB c p next lane a && evalB c p next lane b

/-- Each offset is separately sign-extended, exactly as tt.addptr. -/
def address (c : Config) (p : Nat) (next : I32) (lane : Nat) (ptr : Pointer) : Int :=
  (ptr.offsets.map fun e => (evalI c p next lane e).toInt).foldl (· + ·) 0

structure Plan where
  start : Nat
  stop : Nat
  reducer : BExpr
  nextInitial : IExpr
  continues : BExpr
  poll : Pointer
  load : Pointer
  advance : IExpr
  publish : Pointer
  release : Pointer
  output : Pointer
  mask : BExpr
  deriving DecidableEq, BEq, Repr

/-- Compile this task's full source control schema. This is a restricted source
compiler, not a general Python interpreter. Every source statement is matched;
extra/reordered instructions or unsupported operations fail closed. The type
of each memory/float instruction is explicit. Nonzero CAS replacements and
other floating operators are rejected, rather than interpreted as this kernel.
-/
def compile : List Statement → Option Plan
  | [.integer "pid" .pid,
     .integer "pid_m" (.sdiv .pid (.param .numSms)),
     .integer "pid_n" (.srem .pid (.param .numSms)),
     .zeroFP32 "acc" .bm .bn,
     .forRange "iters" start stop
       [.branch reducer
         [.integer "next_pid" nextInitial,
          .whileLoop (.pure continues)
            [.whileLoop (.casNe poll (.literal 1) (.literal 1) (.literal 1)) [.pass],
             .range "rm1" .bm, .range "rn1" .bn,
             .pointer "P_" pLoad,
             .loadFP32 "acc1" pLoad', .addFP32 "acc" "acc1",
             .integer "next_pid" advance]]
         [.range "rm1" .bm, .range "rn1" .bn,
          .pointer "P_" pStore,
          .storeFP32 pStore' "acc" none,
          .xchg release (.literal 1)],
        .integer "rm" rm, .integer "rn" rn, .pointer "C_" out,
        .mask "mask" mask, .storeFP32 out' "acc" (some mask')]] =>
    if pLoad == pLoad' && pStore == pStore' && out == out' && mask == mask' &&
       poll.region == .locks && release.region == .locks &&
       pLoad.region == .partials && pStore.region == .partials && out.region == .output &&
       out.offsets == [.mul rm (.param .strideM), .mul rn (.param .strideN)] &&
       mask == .and (.lt rm (.param .m)) (.lt rn (.param .n)) then
      some ⟨start, stop, reducer, nextInitial, continues, poll, pLoad, advance,
        pStore, release, out, mask⟩
    else none
  | _ => none

/-- Execute the compiled control plan. Conditions and every address are read
from the source plan rather than copied from Config's convenience functions.
The compiler has already checked fp32 zero/add/load/store and CAS(1,1).
-/
def decodePlan (plan : Plan) (c : Config) (p : Nat) (s : State V) : Event × Control :=
  let ctl := s.control p
  if p ≥ c.grid then (.silent, ctl) else
  match ctl.phase with
  | .branch =>
    if ctl.round ≥ plan.stop then (.silent, { ctl with phase := .done }) else
    if !c.divisionDefined p then (.silent, { ctl with phase := .fault }) else
    (.silent, { ctl with phase := if evalB c p 0 0 plan.reducer then .scan (evalI c p 0 0 plan.nextInitial) else .publish 0 })
  | .scan next =>
    (.silent, { ctl with phase := if evalB c p next 0 plan.continues then .poll next else .output 0 })
  | .poll next =>
    let a := address c p next 0 plan.poll
    (.poll a, { ctl with phase := if s.mem.locks a == 1 then .gather next 0 else .poll next })
  | .gather next lane =>
    if lane < c.lanes then
      (.add p lane (address c p next lane plan.load), { ctl with phase := .gather next (lane + 1) })
    else (.silent, { ctl with phase := .scan (evalI c p next 0 plan.advance) })
  | .publish lane =>
    if lane < c.lanes then
      (.publish p lane (address c p 0 lane plan.publish), { ctl with phase := .publish (lane + 1) })
    else (.silent, { ctl with phase := .release })
  | .release => (.release (address c p 0 0 plan.release), { ctl with phase := .output 0 })
  | .output lane =>
    if lane < c.lanes then
      (if evalB c p 0 lane plan.mask then .output p lane (address c p 0 lane plan.output) else .silent,
       { ctl with phase := .output (lane + 1) })
    else (.silent, { round := ctl.round + 1, phase := .branch })
  | .done | .fault => (.silent, ctl)

def initialPlan (plan : Plan) (b : Backend V) (mem : Memory V) : State V :=
  ⟨mem, fun _ => { round := plan.start }, fun _ _ => b.zero⟩

def stepPlan (plan : Plan) (b : Backend V) (c : Config) (p : Nat) (s : State V) : Event × State V :=
  let (e, ctl) := decodePlan plan c p s
  (e, { execute b e s with control := put s.control p ctl })

def runPlan (plan : Plan) (b : Backend V) (c : Config) : List Nat → State V → List Event × State V
  | [], s => ([], s)
  | p :: ps, s =>
    let (e, s') := stepPlan plan b c p s
    let (es, final) := runPlan plan b c ps s'
    (e :: es, final)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.Frontend
