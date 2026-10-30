import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReady
import bench.tritonbench_g.fused_recurrent_retention.Proofs.StagedReductions

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section

/-- Actual source scalar parameters, including independently strided read-only
cotangents. Forward q/k/v retain their source head strides and DK/DV advances. -/
def layout (s : Shape) (b : InputBuffers) : Layout :=
  ⟨b.q.strides 1,b.q.strides 2,b.q.strides 3,
   b.v.strides 1,b.v.strides 2,b.v.strides 3,s.B,s.H,s.T,s.BK,s.BV,s.DK,s.DV,
   b.do_.strides 0,b.do_.strides 1,b.do_.strides 2,b.do_.strides 3,
   b.finalGrad.strides 0,b.finalGrad.strides 1,b.finalGrad.strides 2,b.finalGrad.strides 3⟩

def forwardRegions : ForwardRegions := ⟨"q","k","v","o","initial_state","final_state"⟩
def backwardRegions : BackwardRegions := ⟨"q","k","v","do","dq","dk","dv","initial_state","d_final_state"⟩

def launch (u : State α) : ProgramId := ⟨u.pids 0,u.pids 1,u.pids 2⟩

def keyOffset (p : Layout) (pid : ProgramId) (t lane : Nat) : Nat :=
  pid.head*p.qHead+pid.ik*p.BK+lane+t*p.DK

def valueOffset (p : Layout) (pid : ProgramId) (t lane : Nat) : Nat :=
  pid.head*p.vHead+pid.iv*p.BV+lane+t*p.DV

def outputOffset (p : Layout) (pid : ProgramId) (t lane : Nat) : Nat :=
  (pid.head+pid.ik*p.B*p.H)*p.vHead+pid.iv*p.BV+lane+t*p.DV

def seedOffset (p : Layout) (pid : ProgramId) (key value : Nat) : Nat :=
  pid.head*p.DK*p.DV+(pid.ik*p.BK+key)*p.DV+(pid.iv*p.BV+value)

def readWord [Inhabited α] (u : State (Staged.Node α)) (r : RegionName) (offset : Nat) : α :=
  ((u.mem r offset).read .real).value

/-- Inputs are ordinary initial-memory observations; this does not define them
through execution, recurrence, output buffers or mathematical postconditions. -/
def inputs [Inhabited α] (p : Layout) (u : State (Staged.Node α)) : Inputs α where
  q := fun h t k => readWord u "q" (h*p.qHead+k+t*p.DK)
  k := fun h t k => readWord u "k" (h*p.qHead+k+t*p.DK)
  v := fun h t v => readWord u "v" (h*p.vHead+v+t*p.DV)
  do_ := fun h t v => readWord u "do"
    ((h/p.H)*p.doBatch+(h%p.H)*p.doHead+v*p.doChannel+t*p.doTime)
  seed := fun h k v => readWord u "initial_state" (h*p.DK*p.DV+k*p.DV+v)
  finalGrad := fun h k v => readWord u "d_final_state"
    ((h/p.H)*p.dfBatch+(h%p.H)*p.dfHead+k*p.dfKey+v*p.dfValue)

def carryNode (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) : Staged.Node α :=
  Staged.tagged (if t=0 then .plain else .forwardState)
    (programForwardCarry a s x seed pid t key value)

@[simp] theorem carryNode_value (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) :
    (carryNode a s x seed pid t key value).value = programForwardCarry a s x seed pid t key value := rfl

@[simp] theorem carryNode_succ (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (pid : ProgramId) (t key value : Nat) :
    carryNode a s x seed pid (t+1) key value =
      Staged.tagged .forwardState (a.fma (decay a (pid.head%s.H))
        (programForwardCarry a s x seed pid t key value)
        (a.mul (keyLoad a s x.k pid.head t (pid.key s key))
          (valueLoad a s x.v pid.head t (pid.value s value)))) := by
  simp [carryNode,programForwardCarry,foldN]

structure ForwardInvariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (u : State (Staged.Node α)) : Prop where
  scale : u.regs .real [] "scale" = some (fun _ => Staged.tagged .scale (a.scale s.DK))
  keyId : u.regs .nat [] "i_k" = some (fun _ => pid.ik)
  valueId : u.regs .nat [] "i_v" = some (fun _ => pid.iv)
  headId : u.regs .nat [] "i_bh" = some (fun _ => pid.head)
  beta : u.regs .real [] "b_b" = some (fun _ => Staged.tagged .beta (decay a (pid.head%s.H)))
  q : u.regs .ptr [s.BK] "p_q" = some (fun i : TileIndex [s.BK] => (("q" : RegionName),keyOffset (layout s b) pid t i.1.val))
  k : u.regs .ptr [s.BK] "p_k" = some (fun i : TileIndex [s.BK] => (("k" : RegionName),keyOffset (layout s b) pid t i.1.val))
  v : u.regs .ptr [s.BV] "p_v" = some (fun i : TileIndex [s.BV] => (("v" : RegionName),valueOffset (layout s b) pid t i.1.val))
  o : u.regs .ptr [s.BV] "p_o" = some (fun i : TileIndex [s.BV] => (("o" : RegionName),outputOffset (layout s b) pid t i.1.val))
  maskK : u.regs .bool [s.BK] "mask_bk" = some (fun i : TileIndex [s.BK] => decide (pid.key s i.1.val<s.DK))
  maskV : u.regs .bool [s.BV] "mask_bv" = some (fun i : TileIndex [s.BV] => decide (pid.value s i.1.val<s.DV))
  maskKV : u.regs .bool [s.BV,s.BK] "mask_kv" =
    some (fun i : TileIndex [s.BV,s.BK] => decide (pid.key s i.2.1.val<s.DK ∧ pid.value s i.1.val<s.DV))
  carry : u.regs .real [s.BV,s.BK] "h" = some (fun i : TileIndex [s.BV,s.BK] =>
    carryNode a s (inputs (layout s b) initial) seed pid t i.2.1.val i.1.val)
  qmem : u.mem "q" = initial.mem "q"
  kmem : u.mem "k" = initial.mem "k"
  vmem : u.mem "v" = initial.mem "v"

#axiomsClean carryNode_succ
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
