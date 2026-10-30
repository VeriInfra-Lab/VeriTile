import bench.tritonbench_g.chunk_delta_fwd.Proofs.NativeGraph
import bench.tritonbench_g.chunk_delta_fwd.Contract

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeReference
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph ieeeDot)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Independent logical input readers. No execution state, output buffer,
source-loop summary or output equality occurs in this interface. -/
structure Inputs where
  (key weight value seed : Nat → Nat → Nat → BitVec 32)

def inputs (p : Parameters) (bits : RegionName → Nat → BitVec 32) : Inputs where
  key := fun head t k => bits "key" (head*(p.T*p.K)+k+t*p.K)
  weight := fun head t k => bits "weight"
    (head/p.H*p.wBatch+head%p.H*p.wHead+t*p.wTime+k*p.wDim)
  value := fun head t v => bits "value" (head*(p.T*p.V)+t*p.V+v)
  seed := fun head k v => bits "initial" (head*p.K*p.V+k*p.V+v)

def zero : Expr := .literal (some .fp32) .real 0
def cast32 (x : Expr) : Expr := .cast (some .fp32) .real .real x
def loaded (active : Prop) [Decidable active] (word : BitVec 32) : Expr :=
  .load (if active then .bits word else zero)
abbrev State (p : Parameters) := Fin p.BK → Fin p.BV → Expr

def seed (p : Parameters) (xs : Inputs) (head tile : Nat) : State p := fun e j =>
  if p.useSeed then cast32 (if e.val<p.K ∧ tile*p.BV+j.val<p.V then
    .bits (xs.seed head e.val (tile*p.BV+j.val)) else zero) else zero

def key (p : Parameters) (xs : Inputs) (head c b : Nat) (e : Fin p.BK) (q : Fin p.BC) : Expr :=
  loaded (e.val<p.K ∧ c*p.BT+b*p.BC+q.val<p.T)
    (xs.key head (c*p.BT+b*p.BC+q.val) e.val)
def weight (p : Parameters) (xs : Inputs) (head c b : Nat) (q : Fin p.BC) (e : Fin p.BK) : Expr :=
  loaded (c*p.BT+b*p.BC+q.val<p.T ∧ e.val<p.K)
    (xs.weight head (c*p.BT+b*p.BC+q.val) e.val)
def value (p : Parameters) (xs : Inputs) (head tile c b : Nat) (q : Fin p.BC) (j : Fin p.BV) : Expr :=
  loaded (c*p.BT+b*p.BC+q.val<p.T ∧ tile*p.BV+j.val<p.V)
    (xs.value head (c*p.BT+b*p.BC+q.val) (tile*p.BV+j.val))

/-- FP32 subtraction from a zero-seeded, ordered BK-feature correction dot.
The physical padded lanes are retained for all FP32 words, including NaN/Inf. -/
def correction (p : Parameters) (xs : Inputs) (head tile c b : Nat)
    (frozen : State p) (q : Fin p.BC) (j : Fin p.BV) : Expr :=
  .binary (some .fp32) .real .sub (value p xs head tile c b q j)
    (.pendingDot (some .fp32) .real .real .real {allowTF32:=some Bool.false}
      p.BC p.BK p.BV q.val j.val
      (fun e => weight p xs head c b q e) (fun e => cast32 (frozen e j)))

def contribution (p : Parameters) (xs : Inputs) (head tile c b : Nat)
    (frozen : State p) (e : Fin p.BK) (j : Fin p.BV) : Expr :=
  .pendingDot (some .fp32) .real .real .real {allowTF32:=some Bool.false}
    p.BK p.BC p.BV e.val j.val (fun q => key p xs head c b e q)
    (fun q => cast32 (correction p xs head tile c b frozen q j))

def accumulate (p : Parameters) (xs : Inputs) (head tile c : Nat) (frozen : State p) :
    Nat → State p
  | 0 => fun _ _ => zero
  | b+1 => fun e j => .binary (some .fp32) .real .add
      (accumulate p xs head tile c frozen b e j) (contribution p xs head tile c b frozen e j)

def advance (p : Parameters) (xs : Inputs) (head tile c : Nat) (frozen : State p) : State p :=
  fun e j => .binary (some .fp32) .real .add (frozen e j)
    (accumulate p xs head tile c frozen ((p.BT+p.BC-1)/p.BC) e j)

/-- Independent rounded-window recurrence on input readers. Staged nodes retain
the one-subblock carry fusion before scalar instruction expansion. -/
def historyState (p : Parameters) (xs : Inputs) (head tile : Nat) : Nat → State p
  | 0 => seed p xs head tile
  | c+1 => advance p xs head tile c (historyState p xs head tile c)

def history (p : Parameters) (h : ChunkDeltaFwd.Geometry.ShapeValid p) (xs : Inputs)
    (head c k v : Nat) (hk : k<p.K) : Graph :=
  ChunkDeltaFwd.Native.lower (cast32
    (historyState p xs head (v/p.BV) c ⟨k,lt_of_lt_of_le hk h.k_cover⟩
      ⟨v%p.BV,Nat.mod_lt _ h.bv_pos⟩))

def corrected (p : Parameters) (h : ChunkDeltaFwd.Geometry.ShapeValid p) (xs : Inputs)
    (head t v : Nat) : Graph :=
  ChunkDeltaFwd.Native.lower (cast32
    (correction p xs head (v/p.BV) (t/p.BT) ((t%p.BT)/p.BC)
      (historyState p xs head (v/p.BV) (t/p.BT))
      ⟨(t%p.BT)%p.BC,Nat.mod_lt _ h.bc_pos⟩ ⟨v%p.BV,Nat.mod_lt _ h.bv_pos⟩))

/-- Every pending dot in the reference is expanded to one scalar FMA per
physical reduction feature; no whole-dot interpretation is an assumption. -/
theorem contribution_scalar (p : Parameters) (xs : Inputs) (head tile c b : Nat)
    (frozen : State p) (e : Fin p.BK) (j : Fin p.BV) :
    ChunkDeltaFwd.Native.lower (contribution p xs head tile c b frozen e j)=
      ieeeDot p.BC (fun q => ChunkDeltaFwd.Native.lower (key p xs head c b e q))
        (fun q => ChunkDeltaFwd.Native.lower (cast32 (correction p xs head tile c b frozen q j)))
        VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero := rfl

theorem correction_scalar (p : Parameters) (xs : Inputs) (head tile c b : Nat)
    (frozen : State p) (q : Fin p.BC) (j : Fin p.BV) :
    ChunkDeltaFwd.Native.lower (correction p xs head tile c b frozen q j)=
      .binary (some .fp32) .real .sub
        (ChunkDeltaFwd.Native.lower (value p xs head tile c b q j))
        (ieeeDot p.BK (fun e => ChunkDeltaFwd.Native.lower (weight p xs head c b q e))
          (fun e => ChunkDeltaFwd.Native.lower (cast32 (frozen e j)))
          VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero) := rfl

/-- A single subblock seeds its update FMA chain with the live old state. -/
theorem one_subblock_native (p : Parameters) (xs : Inputs) (head tile c : Nat)
    (frozen : State p) (e : Fin p.BK) (j : Fin p.BV) :
    ChunkDeltaFwd.Native.lower (.binary (some .fp32) .real .add (frozen e j)
      (accumulate p xs head tile c frozen 1 e j))=
      ieeeDot p.BC (fun q => ChunkDeltaFwd.Native.lower (key p xs head c 0 e q))
        (fun q => ChunkDeltaFwd.Native.lower (cast32 (correction p xs head tile c 0 frozen q j)))
        (ChunkDeltaFwd.Native.lower (frozen e j)) :=
  ChunkDeltaFwd.Native.single_step_fusion _ _ _ _ _ _ _ _ _ _ _ _

/-- Two or more subblocks keep the final outer FP32 addition separate. -/
theorem multiple_subblocks_native (p : Parameters) (xs : Inputs) (head tile c n : Nat)
    (frozen : State p) (e : Fin p.BK) (j : Fin p.BV) :
    ChunkDeltaFwd.Native.lower (.binary (some .fp32) .real .add (frozen e j)
      (accumulate p xs head tile c frozen (n+2) e j))=
      .binary (some .fp32) .real .add (ChunkDeltaFwd.Native.lower (frozen e j))
        (ChunkDeltaFwd.Native.lower (accumulate p xs head tile c frozen (n+2) e j)) := rfl

#axiomsClean contribution_scalar
#axiomsClean correction_scalar
#axiomsClean one_subblock_native
#axiomsClean multiple_subblocks_native
#specNonCircular history avoiding [FP.Structural.run,FP.Structural.exec,
  ChunkDeltaFwd.runGrid,ChunkDeltaFwd.gridMemory,ChunkDeltaFwd.states,ChunkDeltaFwd.historyValue]
#specNonCircular corrected avoiding [FP.Structural.run,FP.Structural.exec,
  ChunkDeltaFwd.runGrid,ChunkDeltaFwd.gridMemory,ChunkDeltaFwd.states,ChunkDeltaFwd.correctedValue]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeReference
