import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Input

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FirstPass
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport Primitive Input
set_option maxHeartbeats 4000000
noncomputable section

def rowInput (s : BlockState) (X : RegionName) (N : Nat) (j : Nat) : WithBot ℝ :=
  s.readMem X (s.pid * N + j)

def maxima (s : BlockState) (X : RegionName) (N B count : Nat) : Tile .real [B] :=
  ⟨fun i => (Recurrence.laneAfter Recurrence.realOperations (rowInput s X N) N B i.1 count).1⟩

def masses (s : BlockState) (X : RegionName) (N B count : Nat) : Tile .real [B] :=
  ⟨fun i => (Recurrence.laneAfter Recurrence.realOperations (rowInput s X N) N B i.1 count).2⟩

structure Stable (N B : Nat) (initial current : BlockState) : Prop where
  pids : current.pids = initial.pids
  numPids : current.numPids = initial.numPids
  mem : current.mem = initial.mem
  undef : current.undef = initial.undef
  pidReg : current.regs .int [] "pid_m" = some (Tile.scalar (initial.pid : Int))
  colsReg : current.regs .int [] "cols" = some (Tile.scalar (N : Int))
  previousReg : current.regs .int [] "prev_multiple" = some (Tile.scalar (Config.previous N B : Int))

structure Invariant (X : RegionName) (N B : Nat) (initial : BlockState)
    (count : Nat) (current : BlockState) : Prop extends Stable N B initial current where
  m : current.regs .real [B] "m" = some (maxima initial X N B count)
  z : current.regs .real [B] "z" = some (masses initial X N B count)

theorem Stable.readMem {N B : Nat} {initial current : BlockState}
    (h : Stable N B initial current) (X : RegionName) (offset : Nat) :
    current.readMem X offset = initial.readMem X offset := by
  unfold BlockState.readMem
  rw [h.mem]

theorem stable_full {N B : Nat} {initial current : BlockState}
    (h : Stable N B initial current) (X : RegionName) (start : Nat) (m z : Tile .real [B]) :
    Stable N B initial (fullState current X N B start initial.pid m z) := by
  constructor
  · exact h.pids
  · exact h.numPids
  · exact h.mem
  · exact h.undef
  · simpa only [fullState, updateState, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.pidReg
  · simpa only [fullState, updateState, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.colsReg
  · simpa only [fullState, updateState, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.previousReg

theorem stable_tail {N B : Nat} {initial current : BlockState}
    (h : Stable N B initial current) (X : RegionName) (start : Nat) (m z : Tile .real [B]) :
    Stable N B initial (tailState current X N B start initial.pid m z) := by
  constructor
  · exact h.pids
  · exact h.numPids
  · exact h.mem
  · exact h.undef
  · simpa only [tailState, updateState, maskedPrefix, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.pidReg
  · simpa only [tailState, updateState, maskedPrefix, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.colsReg
  · simpa only [tailState, updateState, maskedPrefix, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using h.previousReg

theorem full_advance {N B : Nat} {initial current : BlockState}
    (h : Stable N B initial current) (X : RegionName) (count : Nat)
    (hN : 0 < N) (hc : count < Config.fullTiles N B) :
    newMaximum (maxima initial X N B count) (fullLoad current X N B (count * B) initial.pid) =
      maxima initial X N B (count + 1) ∧
    newDenominator (maxima initial X N B count) (masses initial X N B count)
      (fullLoad current X N B (count * B) initial.pid) = masses initial X N B (count + 1) := by
  have hi (i : TileIndex [B]) :
      (fullLoad current X N B (count * B) initial.pid).data i =
      Recurrence.laneInput Recurrence.realOperations (rowInput initial X N) N B count i.1 := by
    unfold fullLoad Recurrence.laneInput rowInput
    simp only [Tile.mk]
    rw [if_pos (Config.full_tile_active hN hc i.1.isLt), h.readMem]
  have hp (i : TileIndex [B]) := update_values
    (maxima initial X N B count) (masses initial X N B count)
    (fullLoad current X N B (count * B) initial.pid) i
  have he (i : TileIndex [B]) :
      ((newMaximum (maxima initial X N B count) (fullLoad current X N B (count * B) initial.pid)).data i,
        (newDenominator (maxima initial X N B count) (masses initial X N B count)
          (fullLoad current X N B (count * B) initial.pid)).data i) =
      ((maxima initial X N B (count + 1)).data i, (masses initial X N B (count + 1)).data i) := by
    rw [hp, hi]
    simp only [maxima, masses, Recurrence.laneAfter_succ]
  constructor
  · apply Tile.ext; intro i; exact congrArg Prod.fst (he i)
  · apply Tile.ext; intro i; exact congrArg Prod.snd (he i)

theorem tail_advance {N B : Nat} {initial current : BlockState}
    (h : Stable N B initial current) (X : RegionName) (count : Nat) :
    newMaximum (maxima initial X N B count) (tailLoad current X N B (count * B) initial.pid) =
      maxima initial X N B (count + 1) ∧
    newDenominator (maxima initial X N B count) (masses initial X N B count)
      (tailLoad current X N B (count * B) initial.pid) = masses initial X N B (count + 1) := by
  have hi (i : TileIndex [B]) :
      (tailLoad current X N B (count * B) initial.pid).data i =
      Recurrence.laneInput Recurrence.realOperations (rowInput initial X N) N B count i.1 := by
    unfold tailLoad Recurrence.laneInput rowInput
    simp only [Tile.mk]
    split <;> first | rfl | rw [h.readMem]
  have hp (i : TileIndex [B]) := update_values
    (maxima initial X N B count) (masses initial X N B count)
    (tailLoad current X N B (count * B) initial.pid) i
  have he (i : TileIndex [B]) :
      ((newMaximum (maxima initial X N B count) (tailLoad current X N B (count * B) initial.pid)).data i,
        (newDenominator (maxima initial X N B count) (masses initial X N B count)
          (tailLoad current X N B (count * B) initial.pid)).data i) =
      ((maxima initial X N B (count + 1)).data i, (masses initial X N B (count + 1)).data i) := by
    rw [hp, hi]
    simp only [maxima, masses, Recurrence.laneAfter_succ]
  constructor
  · apply Tile.ext; intro i; exact congrArg Prod.fst (he i)
  · apply Tile.ext; intro i; exact congrArg Prod.snd (he i)

theorem full_step {N B count : Nat} {initial current : BlockState} (X : RegionName)
    (h : Invariant X N B initial count current) (hN : 0 < N) (hc : count < Config.fullTiles N B) :
    ∃ next, stepStmts (Source.inputFull X N B)
      (current.setReg "start_n" .int [] (Tile.scalar ((count * B : Nat) : Int))) = some next ∧
      Invariant X N B initial (count + 1) next := by
  refine ⟨fullState current X N B (count * B) initial.pid
    (maxima initial X N B count) (masses initial X N B count),
    full_run _ _ _ _ _ _ _ _ h.pidReg h.colsReg h.m h.z, ?_⟩
  have hv := full_advance h.toStable X count hN hc
  refine ⟨stable_full h.toStable _ _ _ _, ?_, ?_⟩
  · simp only [fullState, updateState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true, hv.1]
  · simp only [fullState, updateState, BlockState.setReg_same, hv.2]

theorem tail_step {N B count : Nat} {initial current : BlockState} (X : RegionName)
    (h : Invariant X N B initial count current) :
    ∃ next, stepStmts (Source.inputTail X N B)
      (current.setReg "start_n" .int [] (Tile.scalar ((count * B : Nat) : Int))) = some next ∧
      Invariant X N B initial (count + 1) next := by
  refine ⟨tailState current X N B (count * B) initial.pid
    (maxima initial X N B count) (masses initial X N B count),
    tail_run _ _ _ _ _ _ _ _ h.pidReg h.colsReg h.m h.z, ?_⟩
  have hv := tail_advance h.toStable X count
  refine ⟨stable_tail h.toStable _ _ _ _, ?_, ?_⟩
  · simp only [tailState, updateState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true, hv.1]
  · simp only [tailState, updateState, BlockState.setReg_same, hv.2]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FirstPass
