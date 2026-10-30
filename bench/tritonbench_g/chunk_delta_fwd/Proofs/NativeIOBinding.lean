import bench.tritonbench_g.chunk_delta_fwd.IO
import bench.tritonbench_g.chunk_delta_fwd.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeIOBinding
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface Reference IOBinding
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem key_load (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c b : Nat) (i : TileIndex [p.BK,p.BC]) :
    keyLoad Native.algebra p (tile,head) m c b i =
      NativeReference.key p (nativeInputs p xs) head c b i.1 i.2.1 := by
  by_cases h : i.1.val < p.K ∧ time p c b i.2.1.val < p.T
  all_goals dsimp only [time,column] at h
  · have he := hb.key head (time p c b i.2.1.val) i.1.val hh h.2 h.1 hv
    simpa [keyLoad,NativeReference.key,NativeReference.loaded,nativeInputs,time,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,h] using congrArg Expr.load he
  · simp [keyLoad,NativeReference.key,NativeReference.loaded,time,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,zero,NativeReference.zero,
      show ¬(i.1.val < p.K ∧ c*p.BT+b*p.BC+i.2.1.val < p.T) from h,h]

theorem weight_load (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c b : Nat) (i : TileIndex [p.BC,p.BK]) :
    weightLoad Native.algebra p (tile,head) m c b i =
      NativeReference.weight p (nativeInputs p xs) head c b i.1 i.2.1 := by
  by_cases h : time p c b i.1.val < p.T ∧ i.2.1.val < p.K
  all_goals dsimp only [time,column] at h
  · have he := hb.weight head (time p c b i.1.val) i.2.1.val hh h.1 h.2 hv
    simpa [weightLoad,NativeReference.weight,NativeReference.loaded,nativeInputs,time,weightBase,
      Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,h] using congrArg Expr.load he
  · simp [weightLoad,NativeReference.weight,NativeReference.loaded,time,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,zero,NativeReference.zero,
      show ¬(c*p.BT+b*p.BC+i.1.val < p.T ∧ i.2.1.val < p.K) from h,h]

theorem value_load (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H)
    (tile c b : Nat) (i : CIndex p) :
    valueLoad Native.algebra p (tile,head) m c b i =
      NativeReference.value p (nativeInputs p xs) head tile c b i.1 i.2.1 := by
  by_cases h : time p c b i.1.val < p.T ∧ column p (tile,head) i.2.1.val < p.V
  all_goals dsimp only [time,column] at h
  · have he := hb.value head (time p c b i.1.val) (column p (tile,head) i.2.1.val) hh h.1 h.2
    simpa [valueLoad,NativeReference.value,NativeReference.loaded,nativeInputs,time,column,
      Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,h] using congrArg Expr.load he
  · simp [valueLoad,NativeReference.value,NativeReference.loaded,time,column,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,zero,NativeReference.zero,
      show ¬(c*p.BT+b*p.BC+i.1.val < p.T ∧ tile*p.BV+i.2.1.val < p.V) from h,h]

theorem seed (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H)
    (tile : Nat) (i : HIndex p) :
    initial Native.algebra p (tile,head) m i =
      NativeReference.seed p (nativeInputs p xs) head tile i.1 i.2.1 := by
  cases hon : p.useSeed
  · simp [initial,NativeReference.seed,hon,zero,NativeReference.zero,Native.algebra,
      VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]
  · by_cases h : i.1.val < p.K ∧ tile*p.BV+i.2.1.val < p.V
    · have he := hb.seed hon head i.1.val (tile*p.BV+i.2.1.val) hh h.1 h.2
      simp [initial,NativeReference.seed,hon,column,h,ChunkDeltaFwd.cast,NativeReference.cast32,
        Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra,nativeInputs,he]
    · simp [initial,NativeReference.seed,hon,column,h,ChunkDeltaFwd.cast,NativeReference.cast32,
        zero,NativeReference.zero,Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]

theorem correction (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c b : Nat) (frozen : HValues p Expr) (reference : NativeReference.State p)
    (hf : ∀ i : HIndex p, frozen i = reference i.1 i.2.1) (i : CIndex p) :
    ChunkDeltaFwd.corrected Native.algebra Native.dot p (tile,head) m c b frozen i =
      NativeReference.correction p (nativeInputs p xs) head tile c b reference i.1 i.2.1 := by
  simp only [ChunkDeltaFwd.corrected,Native.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dotValues,toFloat,ofFloat]
  simp only [value_load p xs m hb head hh,weight_load p xs m hb head hh hv]
  simp [NativeReference.correction,ChunkDeltaFwd.cast,NativeReference.cast32,hf,Native.algebra,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]

theorem contribution (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c b : Nat) (frozen acc : HValues p Expr) (reference : NativeReference.State p)
    (hf : ∀ i : HIndex p, frozen i = reference i.1 i.2.1) (i : HIndex p) :
    accumulatedStep Native.algebra Native.dot p (tile,head) m c b frozen acc i =
      .binary (some .fp32) .real .add (acc i)
        (NativeReference.contribution p (nativeInputs p xs) head tile c b reference i.1 i.2.1) := by
  simp only [accumulatedStep,correction p xs m hb head hh hv tile c b frozen reference hf]
  simp only [Native.dot,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dot,
    VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dotValues,toFloat,ofFloat,
    key_load p xs m hb head hh hv]
  simp [NativeReference.contribution,
    ChunkDeltaFwd.cast,NativeReference.cast32,Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]

theorem accumulate (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c : Nat) (frozen : HValues p Expr) (reference : NativeReference.State p)
    (hf : ∀ i : HIndex p, frozen i = reference i.1 i.2.1) (n : Nat) (i : HIndex p) :
    (foldRange (fun b acc => accumulatedStep Native.algebra Native.dot p (tile,head)
      m c b frozen acc) 0 n 1 (fun _ => zero Native.algebra)) i =
      NativeReference.accumulate p (nativeInputs p xs) head tile c reference n i.1 i.2.1 := by
  induction n with
  | zero => simp [foldRange_zero,NativeReference.accumulate,zero,NativeReference.zero,
      Native.algebra,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra]
  | succ n ih =>
    rw [foldRange_succ,contribution p xs m hb head hh hv tile c n frozen _ reference hf,ih]
    rfl

theorem states (p : Parameters) (xs : Inputs (BitVec 32) p) (m : Memory Expr)
    (hb : Reads p Expr.bits xs m) (head : Nat) (hh : head < p.B*p.H) (hv : 0 < p.V)
    (tile c : Nat) (i : HIndex p) :
    ChunkDeltaFwd.states Native.algebra Native.dot p (tile,head) m c i =
      NativeReference.historyState p (nativeInputs p xs) head tile c i.1 i.2.1 := by
  induction c generalizing i with
  | zero => exact seed p xs m hb head hh tile i
  | succ c ih =>
    simp only [ChunkDeltaFwd.states,next,NativeReference.historyState,NativeReference.advance,accumulated]
    rw [accumulate p xs m hb head hh hv tile c _ _ ih,ih]
    rfl

theorem history (p : Parameters) (h : Geometry.ShapeValid p) (xs : Inputs (BitVec 32) p)
    (m : Memory Expr) (hb : Reads p Expr.bits xs m)
    (head c k v : Nat) (hh : head < p.B*p.H) (hk : k < p.K) (hv : v < p.V) :
    Native.lower (historyValue Native.algebra Native.dot p h m head c k v hk) =
      NativeReference.history p h (nativeInputs p xs) head c k v hk := by
  unfold historyValue
  dsimp only [Ownership.pid]
  rw [states p xs m hb head hh (by omega)]
  rfl

theorem corrected (p : Parameters) (h : Geometry.ShapeValid p) (xs : Inputs (BitVec 32) p)
    (m : Memory Expr) (hb : Reads p Expr.bits xs m)
    (head t v : Nat) (hh : head < p.B*p.H) (hv : v < p.V) :
    Native.lower (correctedValue Native.algebra Native.dot p h m head t v) =
      NativeReference.corrected p h (nativeInputs p xs) head t v := by
  unfold correctedValue
  dsimp only [Ownership.pid,Ownership.block]
  rw [correction p xs m hb head hh (by omega) (v/p.BV) (t/p.BT) ((t%p.BT)/p.BC)
    _ _ (states p xs m hb head hh (by omega) (v/p.BV) (t/p.BT))]
  rfl

#axiomsClean history
#axiomsClean corrected
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.NativeIOBinding
