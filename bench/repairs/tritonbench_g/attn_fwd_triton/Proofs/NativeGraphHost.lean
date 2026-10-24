import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeGraph

/-! Lift the actual original word-valued invocation into leaf expressions,
execute the whole selected graph, and evaluate its returned physical cells.
Original inputs, host preparation, output identity and full frames are retained.
No final output, successful execution or numerical equality is a premise. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open TypedExecution TypedHost
open Verified.PhysicalCells (Layout Role)
open Verified.HostSetup (Call calls beforeEmpty nonemptyChecks)
open HostShape (blockD)
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def liftValue : (d : TileDType) → Value Word d → Value Expr d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Expr.leaf
  | .nat | .int | .bool | .ptr | .blockPtr => id

def evaluateValue (ops : Operations) : (d : TileDType) → Value Expr d → Value Word d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => evaluate ops
  | .nat | .int | .bool | .ptr | .blockPtr => id

def liftCell : Cell Word → Cell Expr
  | .mk d v => .mk d (liftValue d v)

def evaluateCell (ops : Operations) : Cell Expr → Cell Word
  | .mk d v => .mk d (evaluateValue ops d v)

@[simp] theorem evaluate_lift (ops : Operations) (c : Cell Word) :
    evaluateCell ops (liftCell c)=c := by
  cases c with | mk d v => cases d <;> rfl

theorem lift_read (c : Cell Word) (d : TileDType) :
    (liftCell c).read d=liftValue d (c.read d) := by
  cases c with | mk tag v => cases tag <;> cases d <;> rfl

def liftPhysical {a} (memory : Physical Word a) : Physical Expr a := fun cell => liftCell (memory cell)

def evaluatePhysical (ops : Operations) {a} (memory : Physical Expr a) : Physical Word a :=
  fun cell => evaluateCell ops (memory cell)

def liftState (s : State Word) : State Expr where
  mem := fun r o => liftCell (s.mem r o)
  regs := fun d sh n => (s.regs d sh n).map (fun xs i => liftValue d (xs i))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r o => liftValue d (s.undef d r o)

def liftRequest (request : Request Word) : Request Expr :=
  { request with value := fun r z h n d => .leaf (request.value r z h n d) }

def leafWord : Expr → Word | .leaf word => word | _ => 0

theorem lift_physical_read (a : Layout) (memory : Physical Word a.toLayout) (k : AliasMemory.Key) :
    TypedPhysical.read a.toLayout (liftPhysical memory) k=
      liftCell (TypedPhysical.read a.toLayout memory k) := by
  unfold TypedPhysical.read
  split <;> rfl

/-- Coordinate-preserving preparation is lifted mechanically from the same
word-valued request and actual original caller cells. Negation acts only on
those input leaves, using the original tensor backend's sign operation. -/
def liftPrepared {request : Request Word} {a : Layout} {caller : Physical Word a.toLayout}
    (ret : Prepared request a caller) : Prepared (liftRequest request) a (liftPhysical caller) where
  original := {
    owned := ret.original.owned
    cell := ret.original.cell
    negative := ret.original.negative
    negate := fun x => .leaf (ret.original.negate (leafWord x))
    ownedInputs := ret.original.ownedInputs
    fp32 := by
      intro z h n d r hr ha
      obtain ⟨x,hx⟩ := ret.original.fp32 z h n d r hr ha
      exact ⟨.leaf x,congrArg liftCell hx⟩
    logical := by
      intro z h n d r hr ha
      change Expr.leaf (request.value r z h n d)=_
      rw [ret.original.logical z h n d r hr ha]
      change Expr.leaf (if ret.original.negative r then _ else _) =
        if ret.original.negative r then _ else _
      rw [show (liftPhysical caller (ret.original.cell r z h n d)).read .fp32=
        Expr.leaf ((caller (ret.original.cell r z h n d)).read .fp32) from lift_read _ _]
      cases ret.original.negative r <;> rfl }
  p := ret.p
  shape := ret.shape
  stage := ret.stage
  base := ret.base
  views := ret.views
  returnedOutput := ret.returnedOutput
  outputBinding := ret.outputBinding
  alloc := ret.alloc
  allocated := ret.allocated
  nonoverlap := ret.nonoverlap
  fresh := ret.fresh
  initial := liftPhysical ret.initial
  preparationWrites := ret.preparationWrites
  preparationFrame := fun cell h => congrArg liftCell (ret.preparationFrame cell h)
  preparationFresh := ret.preparationFresh
  outputFresh := ret.outputFresh
  inputs := by
    intro z h n d r hr ha
    rw [lift_physical_read,ret.inputs z h n d r hr ha]
    rfl

def reference (ops : Operations) (p : Config) (BD : Nat) (a : Layout)
    (initial : Physical Word a.toLayout) (template : State Word) (c : Coordinate p) : Cell Word :=
  evaluateCell ops (TypedExecution.reference backend p BD (support BD)
    (pull a.toLayout (liftPhysical initial) (liftState template)) c)

/-- Complete selected native-graph invocation, starting from the actual
original word-valued caller and its ordinary same-request preparation.
The physical reference runs the same division graph on that initial memory. -/
theorem successful_return (ops : Operations)
    (request : Request Word) (a : Layout) (caller : Physical Word a.toLayout)
    (ret : Prepared request a caller) (template : State Word) (ok : Call → Bool)
    (hbefore : calls ok beforeEmpty=.ok ()) (hscales : calls ok nonemptyChecks=.ok ())
    (hcompile : ok .compileAndLaunch=Bool.true)
    (hne : ret.p.batches*ret.p.heads*ret.p.context*ret.p.headDim≠0)
    (hlaunch : NativeLaunch.CudaLaunchable (launchShape ret.p) (blockD ret.p.headDim)) :
    TypedHost.setup ret ok=.launch ret.returnedOutput rfl hne hlaunch ∧ NativeCertificate ret ∧
    (ret.p.context≤274877906816 ∧ ret.p.batches*ret.p.heads≤65535 ∧ blockD ret.p.headDim≤1048576) ∧
    ∃ final, ∃ graphPhysical : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
      Nonempty (GridRun backend ret.p (blockD ret.p.headDim) Bool.false
        (pull a.toLayout (liftPhysical ret.initial) (liftState template)) final) ∧
      pull a.toLayout graphPhysical final=final ∧
      physical=evaluatePhysical ops graphPhysical ∧
      (∀ c : Coordinate ret.p, physical (outputCell a ret.p ret.base ret.views c)=
        reference ops ret.p (blockD ret.p.headDim) a ret.initial template c) ∧
      (∀ cell : Fin a.capacity, ¬ret.preparationWrites cell →
        (∀ k, Window ret.p k.1 k.2 → a.address k≠cell.val) → physical cell=caller cell) ∧
      (∀ cell : Fin a.capacity, ret.original.owned cell → physical cell=caller cell) := by
  obtain ⟨_,_,hb,final,graphPhysical,hg,hpull,hread,hframe,howned⟩ :=
    TypedHost.division_successful_return backend (liftRequest request) a (liftPhysical caller)
      (liftPrepared ret) (liftState template) ok hbefore hscales hcompile hne hlaunch
      (support (blockD ret.p.headDim))
  refine ⟨by simp [TypedHost.setup,hbefore,hscales,hcompile,hne,hlaunch],TypedHost.native_certificate ret,
    hb,final,graphPhysical,evaluatePhysical ops graphPhysical,hg,hpull,rfl,?_,?_,?_⟩
  · intro c
    exact congrArg (evaluateCell ops) (hread c)
  · intro cell hp hw
    change evaluateCell ops (graphPhysical cell)=_
    rw [hframe cell hp hw]
    exact evaluate_lift ops (caller cell)
  · intro cell hc
    change evaluateCell ops (graphPhysical cell)=_
    rw [howned cell hc]
    exact evaluate_lift ops (caller cell)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
