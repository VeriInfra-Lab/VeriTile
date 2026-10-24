import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.NativeReference
import bench.repairs.tritonbench_g.chunk_delta_fwd.Range

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeReference
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

theorem seed_binding (p : Parameters) (bits : RegionName → Nat → BitVec 32)
    (head tile : Nat) (i : HIndex p) :
    initial Native.algebra p (tile,head) (Native.inputs bits) i=
      seed p (inputs p bits) head tile i.1 i.2.1 := by
  simp [initial,seed,inputs,Native.inputs,ChunkDeltaFwd.cast,cast32,ChunkDeltaFwd.zero,zero,
    column,Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,Cell.read_same]

theorem correction_binding (p : Parameters) (bits : RegionName → Nat → BitVec 32)
    (head tile c b : Nat) (frozen : HValues p Expr) (reference : State p)
    (hf : ∀ i : HIndex p,frozen i=reference i.1 i.2.1) (i : CIndex p) :
    ChunkDeltaFwd.corrected Native.algebra Native.dot p (tile,head) (Native.inputs bits) c b frozen i=
      correction p (inputs p bits) head tile c b reference i.1 i.2.1 := by
  simp only [ChunkDeltaFwd.corrected,Native.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dotValues,toFloat,ofFloat]
  simp [correction,value,weight,loaded,valueLoad,weightLoad,inputs,Native.inputs,
    ChunkDeltaFwd.cast,cast32,ChunkDeltaFwd.zero,zero,column,time,weightBase,Native.algebra,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,Cell.read_same,hf]

theorem step_binding (p : Parameters) (bits : RegionName → Nat → BitVec 32)
    (head tile c b : Nat) (frozen acc : HValues p Expr) (reference : State p)
    (hf : ∀ i : HIndex p,frozen i=reference i.1 i.2.1) (i : HIndex p) :
    accumulatedStep Native.algebra Native.dot p (tile,head) (Native.inputs bits) c b frozen acc i=
      .binary (some .fp32) .real .add (acc i)
        (contribution p (inputs p bits) head tile c b reference i.1 i.2.1) := by
  simp only [accumulatedStep]
  simp only [correction_binding p bits head tile c b frozen reference hf]
  simp [Native.dot,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dotValues,toFloat,ofFloat,
    contribution,key,keyLoad,loaded,inputs,Native.inputs,
    ChunkDeltaFwd.cast,cast32,ChunkDeltaFwd.zero,zero,time,Native.algebra,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,Cell.read_same]

theorem accumulated_binding (p : Parameters) (bits : RegionName → Nat → BitVec 32)
    (head tile c : Nat) (frozen : HValues p Expr) (reference : State p)
    (hf : ∀ i : HIndex p,frozen i=reference i.1 i.2.1) (n : Nat) (i : HIndex p) :
    (foldRange (fun b acc => accumulatedStep Native.algebra Native.dot p (tile,head)
      (Native.inputs bits) c b frozen acc) 0 n 1 (fun _ => ChunkDeltaFwd.zero Native.algebra)) i=
      accumulate p (inputs p bits) head tile c reference n i.1 i.2.1 := by
  induction n with
  | zero => simp [foldRange_zero,accumulate,ChunkDeltaFwd.zero,zero,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]
  | succ n ih =>
    rw [foldRange_succ,step_binding p bits head tile c n frozen _ reference hf,ih]
    rfl

theorem states_binding (p : Parameters) (bits : RegionName → Nat → BitVec 32)
    (head tile c : Nat) (i : HIndex p) :
    states Native.algebra Native.dot p (tile,head) (Native.inputs bits) c i=
      historyState p (inputs p bits) head tile c i.1 i.2.1 := by
  induction c generalizing i with
  | zero => exact seed_binding p bits head tile i
  | succ c ih =>
    simp only [states,next,historyState,advance,accumulated]
    rw [accumulated_binding p bits head tile c _ _ ih,ih]
    rfl

theorem history_binding (p : Parameters) (h : Geometry.ShapeValid p)
    (bits : RegionName → Nat → BitVec 32) (head c k v : Nat) (hk : k<p.K) :
    Native.historyGraph p h bits head c k v hk=history p h (inputs p bits) head c k v hk := by
  unfold Native.historyGraph historyValue
  rw [states_binding]
  rfl

theorem corrected_binding (p : Parameters) (h : Geometry.ShapeValid p)
    (bits : RegionName → Nat → BitVec 32) (head t v : Nat) :
    Native.correctedGraph p h bits head t v=corrected p h (inputs p bits) head t v := by
  unfold Native.correctedGraph correctedValue
  dsimp only [Ownership.pid,Ownership.block]
  rw [correction_binding p bits head (v/p.BV) (t/p.BT) ((t%p.BT)/p.BC)
    _ _ (states_binding p bits head (v/p.BV) (t/p.BT))]
  rfl

#axiomsClean seed_binding
#axiomsClean correction_binding
#axiomsClean accumulated_binding
#axiomsClean states_binding
#axiomsClean history_binding
#axiomsClean corrected_binding
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeReference
