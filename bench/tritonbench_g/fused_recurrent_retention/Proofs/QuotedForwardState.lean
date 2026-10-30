import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedValues

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem scatter_other {α ι : Type} (items : List ι) (u : State α)
    (r r' : RegionName) (hne : r'≠r) (offset : ι → Nat) (active : ι → Prop)
    [DecidablePred active] (value : ι → Cell α) :
    (items.foldl (fun v i => if active i then v.write r (offset i) (value i) else v) u).mem r' = u.mem r' := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split <;> funext j <;> simp [State.write,hne]

def forwardBeforeStore [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    State (Staged.Node α) :=
  let x := inputs (layout s b) initial
  let u := u.setReg "_k" .real [s.BK] (fun i =>
    Staged.atom (keyLoad a s x.k pid.head t (pid.key s i.1.val)))
  let u := u.setReg "_v" .real [s.BV] (fun i =>
    Staged.atom (valueLoad a s x.v pid.head t (pid.value s i.1.val)))
  let u := u.setReg "_q" .real [s.BK] (fun i =>
    Staged.tagged .scaledQuery (a.mul (keyLoad a s x.q pid.head t (pid.key s i.1.val)) (a.scale s.DK)))
  let u := u.setReg "h" .real [s.BV,s.BK] (fun i =>
    carryNode a s x seed pid (t+1) i.2.1.val i.1.val)
  let u := u.setReg "_o" .real [s.BV,s.BK] (fun i =>
    Staged.product a .outputDot (programForwardCarry a s x seed pid (t+1) i.2.1.val i.1.val)
      (a.mul (keyLoad a s x.q pid.head t (pid.key s i.2.1.val)) (a.scale s.DK)))
  u.setReg "_o" .real [s.BV] (fun i => Staged.atom (programO a s plans x seed pid t i.1.val))

def forwardScatter [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (u : State (Staged.Node α)) : State (Staged.Node α) :=
  (TileShape.allIndices [s.BV]).foldl (fun v i =>
    if pid.value s i.1.val<s.DV then
      v.write "o" (outputOffset (layout s b) pid t i.1.val)
        (.mk .real (Staged.atom (programO a s plans (inputs (layout s b) initial) seed pid t i.1.val)))
    else v) u

def forwardBodyResult [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    State (Staged.Node α) :=
  let u := u.setReg "_t" .nat [] (fun _ => t)
  let u := forwardScatter a s b plans initial seed pid t
    (forwardBeforeStore a s b plans initial u seed pid t)
  let u := u.setReg "p_q" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("q" : RegionName),keyOffset (layout s b) pid (t+1) i.1.val))
  let u := u.setReg "p_k" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("k" : RegionName),keyOffset (layout s b) pid (t+1) i.1.val))
  let u := u.setReg "p_o" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("o" : RegionName),outputOffset (layout s b) pid (t+1) i.1.val))
  u.setReg "p_v" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("v" : RegionName),valueOffset (layout s b) pid (t+1) i.1.val))

theorem forwardBodyResult_invariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (h : ForwardInvariant a s b initial seed pid t u) :
    ForwardInvariant a s b initial seed pid (t+1)
      (forwardBodyResult a s b plans initial u seed pid t) := by
  constructor <;>
    simp [forwardBodyResult,forwardScatter,forwardBeforeStore,h.scale,h.keyId,h.valueId,
      h.headId,h.beta,h.maskK,h.maskV,h.maskKV,h.qmem,h.kmem,h.vmem]

#axiomsClean scatter_other
#axiomsClean forwardBodyResult_invariant
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
