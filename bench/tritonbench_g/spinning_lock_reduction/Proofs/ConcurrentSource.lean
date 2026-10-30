import VeriTile.Triton.Concurrent.Controller
import bench.tritonbench_g.spinning_lock_reduction.Proofs.SourceBridge

/-! The existing source compiler's extracted plan lowered to shared, closed
control data. Host values specialize parameters; pid, next and lane stay live
per-thread values. This contains no independent source schema/compiler. -/
namespace VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
open VeriTile.Bench.TritonBenchG.SpinningLock
open SourceReader
open VeriTile.Triton

def lowerI (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr) : IntExpr → Concurrent.NumExpr
  | .literal n => .word64 (.literal n)
  | .param p => .word64 (.literal (match p with
      | .numSms => h.numSms | .groupK => h.groupK | .m => h.m | .n => h.n
      | .strideM => h.strideM | .strideN => h.strideN | .bm => h.bm | .bn => h.bn
      | .phase => phaseNumber stage))
  | .variable _ => .literal 0
  | .pid => .word64 .thread
  | .next => .word64 next
  | .row => .word64 (.div lane (.literal h.bn))
  | .col => .word64 (.rem lane (.literal h.bn))
  | .signed x | .unsigned x => lowerI h stage next lane x
  | .add a b => .wordAdd (lowerI h stage next lane a) (lowerI h stage next lane b)
  | .mul a b => .wordMul (lowerI h stage next lane a) (lowerI h stage next lane b)
  | .div a b => if a.isUnsigned || b.isUnsigned then
      .unsignedDiv (lowerI h stage next lane a) (lowerI h stage next lane b)
      else .signedDiv (lowerI h stage next lane a) (lowerI h stage next lane b)
  | .rem a b => if a.isUnsigned || b.isUnsigned then
      .unsignedRem (lowerI h stage next lane a) (lowerI h stage next lane b)
      else .signedRem (lowerI h stage next lane a) (lowerI h stage next lane b)

theorem lowerI_eval (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr)
    (p : Nat) (locals : Nat → Nat) (e : IntExpr) :
    (lowerI h stage next lane e).eval p locals =
      (evalI h stage p (next.eval p locals) (lane.eval p locals) e).toNat := by
  induction e with
  | div a b ia ib | rem a b ia ib =>
      simp only [lowerI, evalI]
      split <;> simp_all [Concurrent.NumExpr.eval]
  | param k => cases k <;> rfl
  | _ => simp_all [lowerI, evalI, Concurrent.NumExpr.eval]

def lowerB (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr) : BoolExpr → Concurrent.Test
  | .variable _ => .constant false
  | .eq a b => .wordEq (lowerI h stage next lane a) (lowerI h stage next lane b)
  | .ne a b => .not (.wordEq (lowerI h stage next lane a) (lowerI h stage next lane b))
  | .lt a b => if a.isUnsigned || b.isUnsigned then
      .unsignedLt (lowerI h stage next lane a) (lowerI h stage next lane b)
      else .signedLt (lowerI h stage next lane a) (lowerI h stage next lane b)
  | .and a b => .and (lowerB h stage next lane a) (lowerB h stage next lane b)
  | .or a b => .or (lowerB h stage next lane a) (lowerB h stage next lane b)
  | .not a => .not (lowerB h stage next lane a)

theorem lowerB_eval (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr)
    (s : Concurrent.State V) (p : Nat) (locals : Nat → Nat) (e : BoolExpr) :
    (lowerB h stage next lane e).eval s p locals =
      evalB h stage p (next.eval p locals) (lane.eval p locals) e := by
  induction e with
  | lt a b =>
      simp only [lowerB, evalB]
      split <;> simp [Concurrent.Test.eval, lowerI_eval]
  | _ => simp_all [lowerB, evalB, Concurrent.Test.eval, lowerI_eval, bne]

def lowerP (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr) : Pointer → Concurrent.AddrExpr
  | .base _ | .variable _ => .literal 0
  | .offset p x => .add (lowerP h stage next lane p) (.signed64 (lowerI h stage next lane x))

theorem lowerP_eval (h : HostArgs) (stage : Phased.Stage) (next lane : Concurrent.NumExpr)
    (p : Nat) (locals : Nat → Nat) (pointer : Pointer) :
    (lowerP h stage next lane pointer).eval p locals =
      address h stage p (next.eval p locals) (lane.eval p locals) pointer := by
  induction pointer <;> simp_all [lowerP, Concurrent.AddrExpr.eval, lowerI_eval, address]

def go (label : Nat) (next lane : Concurrent.NumExpr) : Concurrent.Next :=
  .goto label [(1,next),(2,lane)]

/-- Labels 0..8 are branch/scan/poll/gather/publish/release/output/done/fault.
Locals 0..2 are the existing round/next/lane fields. -/
def controller (plan : Plan) (h : HostArgs) (stage : Phased.Stage) : Concurrent.Controller :=
  let zero : Concurrent.NumExpr := .literal 0
  let next : Concurrent.NumExpr := .local 1
  let lane : Concurrent.NumExpr := .local 2
  let litI := lowerI h stage
  let litB := lowerB h stage
  let ptr := lowerP h stage
  { threads := h.numSms
    enabled := .not (litB zero zero plan.skip)
    entry := 0
    initialLocals := [plan.start,0,0]
    terminal := [7]
    code := [
      .branch (.natLt (.local 0) (.literal plan.stop))
        (.emit .silent (.branch (litB zero zero plan.reducer)
          (go 1 (litI zero zero plan.nextInitial) zero) (go 4 zero zero)))
        (.emit .silent (go 7 zero zero)),
      .emit .silent (.branch (litB next zero plan.continues)
        (go 2 next zero) (go 6 zero zero)),
      .emit (.compareExchange 0 (ptr next zero plan.poll)
          (litI next zero plan.compare) (litI next zero plan.replace) 0)
        (.branch (.returnedEq 0 (litI next zero plan.expected))
          (go 3 next zero) (go 2 next zero)),
      .branch (.natLt lane (.literal (Wide.lanes h)))
        (.emit (.assign lane (.add (.register lane) (.load 0 (ptr next lane plan.load))))
          (go 3 next (.add lane (.literal 1))))
        (.emit .silent (go 1 (litI next lane plan.advance) zero)),
      .branch (.natLt lane (.literal (Wide.lanes h)))
        (.emit (.store 0 (ptr zero lane plan.publish) (.register lane))
          (go 4 zero (.add lane (.literal 1))))
        (.emit .barrier (go 5 zero zero)),
      .emit (.exchange 0 (ptr zero zero plan.release) (litI zero zero plan.releaseWord))
        (go 6 zero zero),
      .branch (.natLt lane (.literal (Wide.lanes h)))
        (.branch (litB zero lane plan.mask)
          (.emit (.store 1 (ptr zero lane plan.output) (.register lane))
            (go 6 zero (.add lane (.literal 1))))
          (.emit .silent (go 6 zero (.add lane (.literal 1)))))
        (.emit .silent (.goto 0 [(0,.add (.local 0) (.literal 1)),(1,zero),(2,zero)])),
      .emit .silent (go 7 zero zero),
      .emit .silent (go 8 zero zero)] }

def backend (b : Backend V) : Concurrent.Backend V :=
  ⟨b.zero,b.add,fun _ _ seed => seed,id⟩

def values (s : Concurrent.State V) : State V :=
  ⟨⟨s.mem 0,s.mem 1,s.words 0⟩,fun _ => {},s.registers⟩

def event (thread : Nat) : SourceEvent → Concurrent.Event
  | .memory (.add p lane a) => .assign p lane (.add (.register p lane) (.load 0 a))
  | .memory (.publish p lane a) => .store 0 a (.register p lane)
  | .memory (.output p lane a) => .store 1 a (.register p lane)
  | .memory (.release a) => .exchange 0 a 1
  | .memory (.poll _) | .memory .silent => .silent
  | .cas a compare replace => .compareExchange 0 a compare replace thread 0
  | .exchange a value => .exchange 0 a value
  | .barrier => .barrier

theorem execute_values (b : Backend V) (p : Nat) (e : SourceEvent) (s : Concurrent.State V) :
    values (Concurrent.execute (backend b) (event p e) s) = executeSource b e (values s) := by
  cases e with
  | memory e => cases e <;> simp [event, Concurrent.execute, values, executeSource, execute,
      backend, Concurrent.Expr.eval, Concurrent.update, VeriTile.Bench.TritonBenchG.SpinningLock.put] <;> rfl
  | cas a compare replace => simp [event, Concurrent.execute, values, executeSource, cas, Concurrent.update, VeriTile.Bench.TritonBenchG.SpinningLock.put] <;> rfl
  | exchange a value => simp [event, Concurrent.execute, values, executeSource, Concurrent.update, VeriTile.Bench.TritonBenchG.SpinningLock.put] <;> rfl
  | barrier => rfl

end VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
