import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Grid
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Native

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Native
open Checked
abbrev Word := VeriTile.Bench.Repairs.RotaryEmb.Native.Word
abbrev Node := VeriTile.Bench.Repairs.RotaryEmb.Native.Node
abbrev Instructions := VeriTile.Bench.Repairs.RotaryEmb.Native.Instructions
abbrev atom := VeriTile.Bench.Repairs.RotaryEmb.Native.atom
abbrev project := VeriTile.Bench.Repairs.RotaryEmb.Native.project
abbrev liftMemory := VeriTile.Bench.Repairs.RotaryEmb.Native.liftMemory

/-- The uncached compiler hoists coefficients as the first operands and
contracts the first product in the addition; the cached variant contracts the
second product. These local provenance rules preserve both observed graphs. -/
def uncachedMul (p : Instructions) (a b : Node) : Node :=
  ⟨p.mul b.value a.value,some (b.value,a.value)⟩

def uncachedAdd (p : Instructions) (a b : Node) : Node :=
  atom (match a.product with
    | some (x,y) => p.fma x y b.value
    | none => p.fma (BitVec.ofNat 32 0x3f800000) a.value b.value)

def ops (p : Instructions) (cached : Bool) : Ops Node :=
  if cached then VeriTile.Bench.Repairs.RotaryEmb.Native.ops p
  else ⟨uncachedMul p,VeriTile.Bench.Repairs.RotaryEmb.Native.subtract p,uncachedAdd p⟩

/-- The selected scalar source graph. Both halves use the same coefficient
words. Only the local mul/neg/fma realization is a numerical boundary. -/
def expected (p : Instructions) (r : Request) (w : Work) (m : Memory Word) (odd : Bool) : Word :=
  let a := m (.other r.device) (r.data w.key w.token w.head (r.dim w.pair false))
  let b := m (.other r.device) (r.data w.key w.token w.head (r.dim w.pair true))
  let c := m (.other r.device) (r.coefficient false w.token w.pair)
  let s := m (.other r.device) (r.coefficient true w.token w.pair)
  if r.cached then
    if odd then p.fma b c (p.mul a s) else p.fma a c (p.neg (p.mul b s))
  else
    if odd then p.fma s a (p.mul c b) else p.fma c a (p.neg (p.mul s b))

theorem value_lowering (p : Instructions) (r : Request) (w : Work) (m : Memory Node) (odd : Bool) :
    component (sourceValues (ops p r.cached) r w m) odd=atom (expected p r w (project m) odd) := by
  cases cached : r.cached <;> cases odd <;> simp [sourceValues,sourcePair,component,ops,cached,expected,uncachedMul,uncachedAdd,project,VeriTile.Bench.Repairs.RotaryEmb.Native.project,VeriTile.Bench.Repairs.RotaryEmb.Native.subtract,atom,VeriTile.Bench.Repairs.RotaryEmb.Native.atom,VeriTile.Bench.Repairs.RotaryEmb.Native.ops,VeriTile.Bench.Repairs.RotaryEmb.Native.multiply,VeriTile.Bench.Repairs.RotaryEmb.Native.add]

def wordStep (p : Instructions) (r : Request) (indices : Indices) (w : Work) (m : Memory Word) : Memory Word :=
  let low := expected p r w m false
  let high := expected p r w m true
  let afterLow := VeriTile.Bench.Repairs.RotaryEmb.Stores.store m ((.other r.device,destination r indices w false),low)
  VeriTile.Bench.Repairs.RotaryEmb.Stores.store afterLow ((.other r.device,destination r indices w true),high)

theorem step_project (p : Instructions) (r : Request) (indices : Indices) (w : Work) (m : Memory Node) :
    project (sourceStep (ops p r.cached) r indices w m)=wordStep p r indices w (project m) := by
  have low := value_lowering p r w m false
  have high := value_lowering p r w m true
  funext region a
  simp only [component,Bool.false_eq_true,↓reduceIte] at low high
  simp only [sourceStep,wordStep,project,VeriTile.Bench.Repairs.RotaryEmb.Native.project,VeriTile.Bench.Repairs.RotaryEmb.Stores.store,
    destination,Prod.mk.injEq]
  split_ifs <;> simp_all [atom,VeriTile.Bench.Repairs.RotaryEmb.Native.atom]

def wordExecute (p : Instructions) (r : Request) (indices : Indices) : List Work → Memory Word → Memory Word
  | [],m => m
  | w::rest,m => wordExecute p r indices rest (wordStep p r indices w m)

theorem execute_project (p : Instructions) (r : Request) (indices : Indices) (jobs : List Work) (m : Memory Node) :
    project (execute (ops p r.cached) r indices jobs m)=wordExecute p r indices jobs (project m) := by
  induction jobs generalizing m with
  | nil => rfl
  | cons w rest ih =>
    change project (execute (ops p r.cached) r indices rest (sourceStep (ops p r.cached) r indices w m))=_
    rw [ih,step_project]
    rfl

theorem word_readback (p : Instructions) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (unique : jobs.Nodup)
    (m : Memory Word) (w : Work) (member : w∈jobs) (odd : Bool) :
    wordExecute p c.request indices jobs m (.other c.request.device) (destination c.request indices w odd)=
      expected p c.request w m odd := by
  have h := congrArg VeriTile.Bench.Repairs.RotaryEmb.Native.Node.value (execute_readback (ops p c.request.cached) c indices accepted jobs (liftMemory m)
    active unique w member odd)
  rw [value_lowering] at h
  change project (execute (ops p c.request.cached) c.request indices jobs (liftMemory m)) _ _=_ at h
  rw [execute_project] at h
  exact h

theorem word_frame (p : Instructions) (r : Request) (indices : Indices) (jobs : List Work)
    (m : Memory Word) (region : Region) (a : Nat) (outside : ∀ w∈jobs,¬Writes r indices w region a) :
    wordExecute p r indices jobs m region a=m region a := by
  have h := congrArg VeriTile.Bench.Repairs.RotaryEmb.Native.Node.value (execute_frame (ops p r.cached) r indices jobs (liftMemory m) region a outside)
  change project (execute (ops p r.cached) r indices jobs (liftMemory m)) region a=m region a at h
  rw [execute_project] at h
  exact h

abbrev Obligations := VeriTile.Bench.Repairs.RotaryEmb.Native.Obligations
#axiomsClean value_lowering
#axiomsClean word_readback
#axiomsClean word_frame
#specNonCircular expected avoiding [sourceValues,sourceStep,execute,wordExecute]
end VeriTile.Bench.Repairs.RotaryEmbNopad.Native
