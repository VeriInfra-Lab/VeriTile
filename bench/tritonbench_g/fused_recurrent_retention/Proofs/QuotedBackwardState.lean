import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardNumbers

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def doOffset (p : Layout) (pid : ProgramId) (t lane : Nat) : Nat :=
  (pid.head/p.H)*p.doBatch+(pid.head%p.H)*p.doHead+(pid.iv*p.BV+lane)*p.doChannel+t*p.doTime

def gradientOffset (p : Layout) (pid : ProgramId) (t lane : Nat) : Nat :=
  (pid.head+pid.iv*p.B*p.H)*p.qHead+pid.ik*p.BK+lane+t*p.DK

structure BackwardFirstInvariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (u : State (Staged.Node α)) : Prop where
  scale : u.regs .real [] "scale" = some (fun _ => Staged.tagged .scale (a.scale s.DK))
  keyId : u.regs .nat [] "i_k" = some (fun _ => pid.ik)
  valueId : u.regs .nat [] "i_v" = some (fun _ => pid.iv)
  headId : u.regs .nat [] "i_bh" = some (fun _ => pid.head)
  localHead : u.regs .nat [] "i_h" = some (fun _ => pid.head%s.H)
  beta : u.regs .real [] "b_b" = some (fun _ => Staged.tagged .beta (decay a (pid.head%s.H)))
  k : u.regs .ptr [s.BK] "p_k" = some (fun i : TileIndex [s.BK] =>
    (("k" : RegionName),keyOffset (layout s b) pid t i.1.val))
  v : u.regs .ptr [s.BV] "p_v" = some (fun i : TileIndex [s.BV] =>
    (("v" : RegionName),valueOffset (layout s b) pid t i.1.val))
  dout : u.regs .ptr [s.BV] "p_do" = some (fun i : TileIndex [s.BV] =>
    (("do" : RegionName),doOffset (layout s b) pid t i.1.val))
  dq : u.regs .ptr [s.BK] "p_dq" = some (fun i : TileIndex [s.BK] =>
    (("dq" : RegionName),gradientOffset (layout s b) pid t i.1.val))
  maskK : u.regs .bool [s.BK] "mask_bk" = some (fun i => decide (pid.key s i.1.val<s.DK))
  maskV : u.regs .bool [s.BV] "mask_bv" = some (fun i => decide (pid.value s i.1.val<s.DV))
  carry : u.regs .real [s.BK,s.BV] "h" = some (fun i =>
    carryNode a s (inputs (layout s b) initial) seed pid t i.1.val i.2.1.val)
  qmem : u.mem "q" = initial.mem "q"
  kmem : u.mem "k" = initial.mem "k"
  vmem : u.mem "v" = initial.mem "v"
  domem : u.mem "do" = initial.mem "do"
  dfmem : u.mem "d_final_state" = initial.mem "d_final_state"

def backwardFirstBeforeStore [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    State (Staged.Node α) :=
  let x := inputs (layout s b) initial
  let u := u.setReg "_k" .real [s.BK] (fun i =>
    Staged.atom (keyLoad a s x.k pid.head t (pid.key s i.1.val)))
  let u := u.setReg "_v" .real [s.BV] (fun i =>
    Staged.atom (valueLoad a s x.v pid.head t (pid.value s i.1.val)))
  let u := u.setReg "_do" .real [s.BV] (fun i =>
    Staged.atom (valueLoad a s x.do_ pid.head t (pid.value s i.1.val)))
  let u := u.setReg "h" .real [s.BK,s.BV] (fun i =>
    carryNode a s x seed pid (t+1) i.1.val i.2.1.val)
  let u := u.setReg "_d_q" .real [s.BK,s.BV] (fun i =>
    Staged.product a .queryDot (programForwardCarry a s x seed pid (t+1) i.1.val i.2.1.val)
      (valueLoad a s x.do_ pid.head t (pid.value s i.2.1.val)))
  u.setReg "d_q" .real [s.BK] (fun i => Staged.tagged .scaledQuery (programQ a s plans x seed pid t i.1.val))

def backwardFirstScatter [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (u : State (Staged.Node α)) : State (Staged.Node α) :=
  (TileShape.allIndices [s.BK]).foldl (fun v i =>
    if pid.key s i.1.val<s.DK then
      v.write "dq" (gradientOffset (layout s b) pid t i.1.val)
        (.mk .real (Staged.atom (programQ a s plans (inputs (layout s b) initial) seed pid t i.1.val)))
    else v) u

def backwardFirstBodyResult [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    State (Staged.Node α) :=
  let u := u.setReg "i" .nat [] (fun _ => t)
  let u := backwardFirstScatter a s b plans initial seed pid t
    (backwardFirstBeforeStore a s b plans initial u seed pid t)
  let u := u.setReg "p_k" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("k" : RegionName),keyOffset (layout s b) pid (t+1) i.1.val))
  let u := u.setReg "p_do" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("do" : RegionName),doOffset (layout s b) pid (t+1) i.1.val))
  let u := u.setReg "p_v" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("v" : RegionName),valueOffset (layout s b) pid (t+1) i.1.val))
  u.setReg "p_dq" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("dq" : RegionName),gradientOffset (layout s b) pid (t+1) i.1.val))

theorem backwardFirstBodyResult_invariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (h : BackwardFirstInvariant a s b initial seed pid t u) :
    BackwardFirstInvariant a s b initial seed pid (t+1)
      (backwardFirstBodyResult a s b plans initial u seed pid t) := by
  constructor <;>
    simp [backwardFirstBodyResult,backwardFirstScatter,backwardFirstBeforeStore,
      h.scale,h.keyId,h.valueId,h.headId,h.localHead,h.beta,h.maskK,h.maskV,
      h.qmem,h.kmem,h.vmem,h.domem,h.dfmem]

#axiomsClean backwardFirstBodyResult_invariant
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
