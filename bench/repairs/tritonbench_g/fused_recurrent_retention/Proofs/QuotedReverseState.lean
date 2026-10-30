import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardState

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- These offsets retain the actual dead pointer updates after the final
iteration. Logical reverse-time addresses are proved separately for r<T. -/
def reverseKeyOffset (p : Layout) (pid : ProgramId) (r lane : Nat) : Nat :=
  keyOffset p pid (p.T-1) lane-r*p.DK

def reverseValueOffset (p : Layout) (pid : ProgramId) (r lane : Nat) : Nat :=
  valueOffset p pid (p.T-1) lane-r*p.DV

def reverseDoOffset (p : Layout) (pid : ProgramId) (r lane : Nat) : Nat :=
  doOffset p pid (p.T-1) lane-r*p.doTime

def reverseGradientOffset (p : Layout) (pid : ProgramId) (r lane : Nat) : Nat :=
  gradientOffset p pid (p.T-1) lane-r*p.DK

def reverseOutputOffset (p : Layout) (pid : ProgramId) (r lane : Nat) : Nat :=
  outputOffset p pid (p.T-1) lane-r*p.DV

theorem reverse_offset (base width T r : Nat) (hr : r<T) :
    base+(T-1)*width-r*width=base+(T-1-r)*width := by
  rw [Nat.add_sub_assoc (Nat.mul_le_mul_right width (show r ≤ T-1 by omega))]
  simp only [Nat.sub_mul]

structure BackwardSecondInvariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat)
    (u : State (Staged.Node α)) : Prop where
  scale : u.regs .real [] "scale" = some (fun _ => Staged.tagged .scale (a.scale s.DK))
  beta : u.regs .real [] "b_b" = some (fun _ => Staged.tagged .beta (decay a (pid.head%s.H)))
  q : u.regs .ptr [s.BK] "p_q" = some (fun i : TileIndex [s.BK] =>
    (("q" : RegionName),reverseKeyOffset (layout s b) pid r i.1.val))
  k : u.regs .ptr [s.BK] "p_k" = some (fun i : TileIndex [s.BK] =>
    (("k" : RegionName),reverseKeyOffset (layout s b) pid r i.1.val))
  v : u.regs .ptr [s.BV] "p_v" = some (fun i : TileIndex [s.BV] =>
    (("v" : RegionName),reverseValueOffset (layout s b) pid r i.1.val))
  dout : u.regs .ptr [s.BV] "p_do" = some (fun i : TileIndex [s.BV] =>
    (("do" : RegionName),reverseDoOffset (layout s b) pid r i.1.val))
  dk : u.regs .ptr [s.BK] "p_dk" = some (fun i : TileIndex [s.BK] =>
    (("dk" : RegionName),reverseGradientOffset (layout s b) pid r i.1.val))
  dv : u.regs .ptr [s.BV] "p_dv" = some (fun i : TileIndex [s.BV] =>
    (("dv" : RegionName),reverseOutputOffset (layout s b) pid r i.1.val))
  maskK : u.regs .bool [s.BK] "mask_bk" = some (fun i => decide (pid.key s i.1.val<s.DK))
  maskV : u.regs .bool [s.BV] "mask_bv" = some (fun i => decide (pid.value s i.1.val<s.DV))
  carry : u.regs .real [s.BK,s.BV] "d_h" = some (fun i =>
    reverseNode a s (inputs (layout s b) initial) final pid r i.1.val i.2.1.val)
  qmem : u.mem "q" = initial.mem "q"
  kmem : u.mem "k" = initial.mem "k"
  vmem : u.mem "v" = initial.mem "v"
  domem : u.mem "do" = initial.mem "do"
  dfmem : u.mem "d_final_state" = initial.mem "d_final_state"

def reverseBeforeStores [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) :
    State (Staged.Node α) :=
  let x := inputs (layout s b) initial
  let t := s.T-1-r
  let u := u.setReg "_do" .real [s.BV] (fun i => Staged.atom (valueLoad a s x.do_ pid.head t (pid.value s i.1.val)))
  let u := u.setReg "_q" .real [s.BK] (fun i =>
    Staged.tagged .scaledQuery (a.mul (keyLoad a s x.q pid.head t (pid.key s i.1.val)) (a.scale s.DK)))
  let u := u.setReg "_k" .real [s.BK] (fun i => Staged.atom (keyLoad a s x.k pid.head t (pid.key s i.1.val)))
  let u := u.setReg "_v" .real [s.BV] (fun i => Staged.atom (valueLoad a s x.v pid.head t (pid.value s i.1.val)))
  let u := u.setReg "d_h" .real [s.BK,s.BV] (fun i =>
    Staged.tagged .reverseState (programAdjoint a s x final pid t i.1.val i.2.1.val))
  let u := u.setReg "d_k" .real [s.BK] (fun i => Staged.atom (programK a s plans x final pid t i.1.val))
  let u := u.setReg "d_v" .real [s.BV] (fun i => Staged.atom (programV a s plans x final pid t i.1.val))
  u.setReg "d_h" .real [s.BK,s.BV] (fun i => reverseNode a s x final pid (r+1) i.1.val i.2.1.val)

def reverseScatter [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat)
    (u : State (Staged.Node α)) : State (Staged.Node α) :=
  let u := (TileShape.allIndices [s.BK]).foldl (fun v i =>
    if pid.key s i.1.val<s.DK then
      v.write "dk" (reverseGradientOffset (layout s b) pid r i.1.val)
        (.mk .real (Staged.atom (programK a s plans (inputs (layout s b) initial) final pid (s.T-1-r) i.1.val)))
    else v) u
  (TileShape.allIndices [s.BV]).foldl (fun v i =>
    if pid.value s i.1.val<s.DV then
      v.write "dv" (reverseOutputOffset (layout s b) pid r i.1.val)
        (.mk .real (Staged.atom (programV a s plans (inputs (layout s b) initial) final pid (s.T-1-r) i.1.val)))
    else v) u

def reverseBodyResult [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) :
    State (Staged.Node α) :=
  let u := u.setReg "_t" .nat [] (fun _ => r)
  let u := reverseScatter a s b plans initial final pid r
    (reverseBeforeStores a s b plans initial u final pid r)
  let u := u.setReg "p_do" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("do" : RegionName),reverseDoOffset (layout s b) pid (r+1) i.1.val))
  let u := u.setReg "p_q" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("q" : RegionName),reverseKeyOffset (layout s b) pid (r+1) i.1.val))
  let u := u.setReg "p_k" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("k" : RegionName),reverseKeyOffset (layout s b) pid (r+1) i.1.val))
  let u := u.setReg "p_v" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("v" : RegionName),reverseValueOffset (layout s b) pid (r+1) i.1.val))
  let u := u.setReg "p_dk" .ptr [s.BK] (fun i : TileIndex [s.BK] => (("dk" : RegionName),reverseGradientOffset (layout s b) pid (r+1) i.1.val))
  u.setReg "p_dv" .ptr [s.BV] (fun i : TileIndex [s.BV] => (("dv" : RegionName),reverseOutputOffset (layout s b) pid (r+1) i.1.val))

theorem reverseBodyResult_invariant [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial u : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat)
    (h : BackwardSecondInvariant a s b initial final pid r u) :
    BackwardSecondInvariant a s b initial final pid (r+1)
      (reverseBodyResult a s b plans initial u final pid r) := by
  constructor <;>
    simp [reverseBodyResult,reverseScatter,reverseBeforeStores,
      h.scale,h.beta,h.maskK,h.maskV,h.qmem,h.kmem,h.vmem,h.domem,h.dfmem]

#axiomsClean reverse_offset
#axiomsClean reverseBodyResult_invariant
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
