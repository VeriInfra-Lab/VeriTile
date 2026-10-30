import bench.tritonbench_g.matrix_reduction.Proofs.GraphBase
import bench.tritonbench_g.matrix_reduction.Proofs.MachineMax
import bench.tritonbench_g.matrix_reduction.Proofs.BinaryConversion

/-! The actual typed source executes with 32-bit register encodings. A binary16
memory cell holds its 16-bit word zero-extended in the carrier; promotion writes
the actual binary32 encoding and the store applies the proved inverse on its
reachable image. This is separate from the earlier origin-representation helper. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.ConcreteCorrect
open VeriTile.Triton FP.Structural TypedSource MachineMax BinaryConversion
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

abbrev Word := BitVec 32

def encode : (storage : Storage) → Bits storage → Word
  | .fp16, x => x.zeroExtend 32
  | .fp32, x => x

def promote : (storage : Storage) → Bits storage → Word
  | .fp16, x => widen x
  | .fp32, x => x

def lower : Storage → Word → Word
  | .fp16, x => (narrow x).zeroExtend 32
  | .fp32, x => x

def loadPromote : Storage → Word → Word
  | .fp16, x => widen (x.truncate 16)
  | .fp32, x => x

@[simp] theorem lower_promote (storage : Storage) (x : Bits storage) :
    lower storage (promote storage x) = encode storage x := by
  cases storage
  · exact congrArg (fun x : BitVec 16 => x.zeroExtend 32) (narrow_widen x)
  · rfl

@[simp] theorem loadPromote_encode (storage : Storage) (x : Bits storage) :
    loadPromote storage (encode storage x) = promote storage x := by
  cases storage
  · simp [loadPromote, encode, promote, BitVec.truncate, BitVec.zeroExtend]
    exact congrArg widen (BitVec.setWidth_eq x)
  · rfl

theorem promote_order (storage : Storage) (x y : Bits storage) :
    rank .fp32 (promote storage x) ≤ rank .fp32 (promote storage y) ↔
      rank storage x ≤ rank storage y := by
  cases storage
  · exact widen_order x y
  · rfl

/-- Every used cast is an explicit bit conversion. Unrelated backend operations
remain arbitrary and are never used by the proved source. -/
def backend (rest : Algebra Word) : Algebra Word :=
  { rest with
    cast := fun p src dst x => match src, dst with
      | .fp16, .real => widen (x.truncate 16)
      | .real, .fp16 => (narrow x).zeroExtend 32
      | .real, .real => x
      | _, _ => rest.cast p src dst x
    fp32Load := id
    reduceMax := fun _ => reduce .fp32 }

def input (storage : Storage) (s : State Word) (X : RegionName)
    (sm sn M N : Nat) (i : Fin M) (j : Fin N) : Word :=
  loadPromote storage (toFloat (channel storage)
    ((s.mem X (i.val * sm + j.val * sn)).read (channel storage).toTileDType))

def output (storage : Storage) (s : State Word) (X : RegionName)
    (sm sn M N : Nat) (i : Fin M) : Word :=
  lower storage (rowMaximum .fp32 (input storage s X sm sn M N i))

def writtenMemory (storage : Storage) (s : State Word) (X Y : RegionName)
    (sm sn M N : Nat) : RegionName → Nat → Cell Word :=
  ((TileShape.allIndices [M]).foldl (fun t i =>
    t.write Y i.1.val (.mk (channel storage).toTileDType
      (ofFloat (channel storage) (output storage s X sm sn M N i.1)))) s).mem

theorem source_run (storage : Storage) (rest : Algebra Word)
    (s : State Word) (X Y : RegionName) (sm sn sy M N : Nat) (hN : 0 < N) :
    ∃ t, FP.Structural.exec (backend rest) (kernel storage X Y sm sn sy M N) s = some t ∧
      t.mem = writtenMemory storage s X Y sm sn M N := by
  cases storage <;>
    simp [kernel, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, backend,
      State.setReg, TileShape.axisDim, hN, BlockPtr.inBounds,
      TileShape.insertAxisIndex, TileShape.eraseAxis,
      output, input, writtenMemory, channel, reduce, lower, loadPromote,
      ofFloat, toFloat, Region.cast, ComputeDType.eraseDType]
  all_goals simp only [fold_memory]
  all_goals rfl

theorem written_readback (storage : Storage) (s : State Word) (X Y : RegionName)
    (sm sn M N : Nat) (i : Fin M) :
    writtenMemory storage s X Y sm sn M N Y i.val =
      .mk (channel storage).toTileDType
        (ofFloat (channel storage) (output storage s X sm sn M N i)) :=
  State.scatter_readback Y (fun k : TileIndex [M] => k.1.val) _ s
    (i, PUnit.unit) (lane_injective M)

theorem written_frame (storage : Storage) (s : State Word) (X Y : RegionName)
    (sm sn M N : Nat) (r : RegionName) (o : Nat) (hm : r ≠ Y ∨ M ≤ o) :
    writtenMemory storage s X Y sm sn M N r o = s.mem r o := by
  apply State.scatter_frame
  rcases hm with hm | hm
  · exact Or.inl hm
  · right
    intro i hi he
    have := i.1.isLt
    omega

/-- Binary16 and binary32 source execution with actual binary32 intermediate
encodings. The independent maximum specification concerns original input bits;
exact promotion preserves order and narrowing recovers the selected original
word, including signed zero. Non-NaN domain includes subnormals and infinities.
Native scheduling/NaN-payload/zero-sign lowering is not asserted. -/
theorem source_correct
    (storage : Storage) (rest : Algebra Word)
    (s : State Word) (X Y : RegionName) (sm sn sy M N : Nat) (hN : 0 < N)
    (xs : Fin M → Fin N → Bits storage)
    (hinput : ∀ i j, s.mem X (i.val * sm + j.val * sn) =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (xs i j))))
    (hnotNaN : ∀ i j, isNaN storage (xs i j) = Bool.false) :
    ∃ t, FP.Structural.exec (backend rest) (kernel storage X Y sm sn sy M N) s = some t ∧
      (∀ i : Fin M, ∃ v : Bits storage,
        t.mem Y i.val = .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage v)) ∧
        IsMaximum storage (xs i) v ∧ isNaN storage v = Bool.false) ∧
      (∀ r o, r ≠ Y ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run storage rest s X Y sm sn sy M N hN
  refine ⟨t, he, ?_, ?_⟩
  · intro i
    have hi : input storage s X sm sn M N i = fun j => promote storage (xs i j) := by
      funext j
      unfold input
      rw [hinput]
      have hr : toFloat (channel storage)
          (Cell.read (channel storage).toTileDType
            (.mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (xs i j))))) =
          encode storage (xs i j) := by cases storage <;> rfl
      rw [hr, loadPromote_encode]
    let j := best .fp32 (fun j => promote storage (xs i j)) hN
    have ho : output storage s X sm sn M N i = encode storage (xs i j) := by
      unfold output
      rw [hi]
      unfold rowMaximum
      rw [dif_pos hN]
      exact lower_promote storage (xs i j)
    refine ⟨xs i j, ?_, ?_, hnotNaN i j⟩
    · rw [hm, written_readback, ho]
    · refine ⟨⟨j, rfl⟩, ?_⟩
      intro k
      exact (promote_order storage (xs i k) (xs i j)).mp
        (best_maximal .fp32 (fun j => promote storage (xs i j)) hN k)
  · intro r o hmiss
    rw [hm]
    exact written_frame storage s X Y sm sn M N r o hmiss

#axiomsClean narrow_widen
#axiomsClean widen_order
#axiomsClean widen_preserves_finite_magnitude
#axiomsClean widen_preserves_sign
#axiomsClean widen_positive_infinity
#axiomsClean widen_negative_infinity
#axiomsClean source_run
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.ConcreteCorrect
