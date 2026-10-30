import bench.tritonbench_g.relu_strided_buffer.Proofs.CappedGrid
import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedView
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostMetadata
import VeriTile.Meta.StatementAudit

/-!
Selected-FP32 metadata and ordered pre-launch stages of the original wrapper.
The same six-shift Triton helper is reused from an existing host transcription.
No kernel result is supplied by the host model. CUDA guard success/failure is an
ordinary runtime boundary; launch execution is supplied separately.

General shapes are retained in the preparation function. Rank-one theorems
specialize the declared wrapper domain without inserting a new Python guard.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostPlan
open VeriTile.Triton
open SignedView

abbrev nextPower := VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo

/-- Metadata of a Tensor or StridedBuffer after selecting FP32 storage. `base`
is an element offset relative to the allocation named by `region`. -/
structure Buffer where
  region : RegionName
  base : Int
  shape : List Nat
  strides : List Int
  deviceIndex : Option Nat

def Buffer.numel (x : Buffer) : Nat := x.shape.prod
def Buffer.byteOffset (x : Buffer) : Int := 4 * x.base
def Buffer.view (x : Buffer) (stride : Int) : View := ⟨x.region, x.base, stride⟩

/-- The constructor keeps the base tensor's device and applies the selected
dtype's itemsize to `offset`; shape and stride overrides remain independent. -/
def reinterpret (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (offset : Int) : Buffer :=
  { base with
    base := base.base + offset
    shape := shape.getD base.shape
    strides := strides.getD base.strides }

theorem reinterpret_bytes (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (offset : Int) :
    (reinterpret base shape strides offset).byteOffset = base.byteOffset + 4 * offset := by
  simp [reinterpret, Buffer.byteOffset, Int.mul_add]

inductive Failure where
  | shapeAssertion | tileDivision | ctaDivision | deviceGuard
  | inputStrideAccess | outputStrideAccess | shapeAccess | tileSizeAccess
  deriving DecidableEq, Repr

/-- Iterate dimensions in the actual reverse order, including the division
after the final tile choice. Nat division by zero is never used as success. -/
def chooseReverse : Nat → List Nat → Except Failure (List Nat)
  | _, [] => .ok []
  | cap, n :: ns =>
    let b := min cap (nextPower n)
    if b = 0 then .error .tileDivision else do
      let bs ← chooseReverse (max 1 (cap / b)) ns
      pure (b :: bs)

def chooseTiles (shape : List Nat) : Except Failure (List Nat) := do
  let bs ← chooseReverse 512 shape.reverse
  pure bs.reverse

def numWarps (tileSize : Nat) : Nat :=
  if tileSize < 2048 then 4 else if tileSize < 4096 then 8 else 16

structure Partition where
  shape : List Nat
  numTasks : Nat
  tileSizes : List Nat
  tileSize : Nat
  numTiles : Nat
  numCTAs : Nat
  tilesPerCTA : Nat
  warps : Nat
  oneTile : Bool
  deriving DecidableEq, Repr

/-- Exact shape-product, zipped cdiv product, cap and branch selection. -/
def partition (shape : List Nat) : Except Failure Partition := do
  let bs ← chooseTiles shape
  let tileSize := bs.prod
  let count := (List.zipWith cdiv shape bs).prod
  let c := min 65536 count
  if c = 0 then .error .ctaDivision else
    let t := cdiv count c
    pure ⟨shape, shape.prod, bs, tileSize, count, c, t,
      numWarps tileSize, decide (t = 1)⟩

structure Arguments where
  input : View
  output : View
  n : Nat
  numTasks : Nat
  B : Nat
  C : Nat
  T : Nat
  warps : Nat
  oneTile : Bool
  deviceIndex : Option Nat

def headOr (xs : List α) (failure : Failure) : Except Failure α :=
  match xs with | [] => .error failure | x :: _ => .ok x

/-- These accesses occur inside the guard, in Python argument evaluation
order. No rank/stride assertion absent from the source is introduced. -/
def arguments (input output : Buffer) (p : Partition) : Except Failure Arguments := do
  let sx ← headOr input.strides .inputStrideAccess
  let sy ← headOr output.strides .outputStrideAccess
  let n ← headOr p.shape .shapeAccess
  let b ← headOr p.tileSizes .tileSizeAccess
  pure ⟨input.view sx, output.view sy, n, p.numTasks, b,
    p.numCTAs, p.tilesPerCTA, p.warps, p.oneTile, input.deviceIndex⟩

/-- Pre-launch source order. `guardSucceeds` is only the fallible CUDA guard
result; it neither computes memory nor assumes a desired kernel result. -/
def prepare (input output : Buffer) (guardSucceeds : Bool) : Except Failure Arguments :=
  if input.shape = output.shape then do
    let p ← partition output.shape
    if guardSucceeds then arguments input output p else .error .deviceGuard
  else .error .shapeAssertion

def block (n : Nat) : Nat := min 512 (nextPower n)

theorem block_pos {n : Nat} (hn : 0 < n) : 0 < block n := by
  exact lt_min (by omega)
    (VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo_pos n hn)

theorem block_le (n : Nat) : block n ≤ 512 := Nat.min_le_left _ _

theorem rank_one_tiles {n : Nat} (hn : 0 < n) : chooseTiles [n] = .ok [block n] := by
  have hb : block n ≠ 0 := Nat.ne_of_gt (block_pos hn)
  simp [chooseTiles, chooseReverse, block] at hb ⊢
  simp [hb]

theorem block_warps (n : Nat) : numWarps (block n) = 4 := by
  have hb := block_le n
  simp [numWarps, show block n < 2048 by omega]

def rankOnePartition (n : Nat) : Partition :=
  let b := block n
  ⟨[n], n, [b], b, CappedGrid.numTiles n b, CappedGrid.numCTAs n b,
    CappedGrid.tilesPerCTA n b, 4, decide (CappedGrid.tilesPerCTA n b = 1)⟩

theorem rank_one_partition {n : Nat} (hn : 0 < n) :
    partition [n] = .ok (rankOnePartition n) := by
  have hc := CappedGrid.numCTAs_pos hn (block_pos hn)
  rw [partition, rank_one_tiles hn]
  simp [bind, Except.bind, pure, Except.pure, rankOnePartition, CappedGrid.numTiles, CappedGrid.numCTAs,
    CappedGrid.tilesPerCTA, block_warps, show min 65536 (cdiv n (block n)) ≠ 0 by
      exact Nat.ne_of_gt hc]

/-- The source fails in the heuristic before any device guard or launch. -/
theorem empty_partition : partition [0] = .error .tileDivision := by
  rfl

theorem shape_assertion_first (input output : Buffer) (guardSucceeds : Bool)
    (h : input.shape ≠ output.shape) :
    prepare input output guardSucceeds = .error .shapeAssertion := by
  simp [prepare, h]

theorem empty_before_guard (input output : Buffer) (guardSucceeds : Bool)
    (hin : input.shape = [0]) (hout : output.shape = [0]) :
    prepare input output guardSucceeds = .error .tileDivision := by
  simp [prepare, hin, hout, empty_partition, bind, Except.bind]

theorem guard_before_stride_access (input output : Buffer) (p : Partition)
    (hshape : input.shape = output.shape) (hp : partition output.shape = .ok p) :
    prepare input output Bool.false = .error .deviceGuard := by
  simp [prepare, hshape, hp, bind, Except.bind]

def rankOneArguments (input output : Buffer) (n : Nat) (sx sy : Int) : Arguments :=
  let b := block n
  ⟨input.view sx, output.view sy, n, n, b,
    CappedGrid.numCTAs n b, CappedGrid.tilesPerCTA n b,
    4, decide (CappedGrid.tilesPerCTA n b = 1), input.deviceIndex⟩

theorem rank_one_prepare (input output : Buffer) {n : Nat} (hn : 0 < n)
    (sx sy : Int) (hin : input.shape = [n]) (hout : output.shape = [n])
    (hsi : input.strides = [sx]) (hso : output.strides = [sy]) :
    prepare input output Bool.true = .ok (rankOneArguments input output n sx sy) := by
  simp [prepare, hin, hout, rank_one_partition hn, arguments, headOr,
    hsi, hso, rankOnePartition, rankOneArguments, bind, Except.bind, pure, Except.pure]

/-- The exact prepared launch parameters instantiate the already checked
capped-grid coverage, for every logical element and both source branches. -/
theorem prepared_coverage (input output : Buffer) {n i : Nat}
    (sx sy : Int) (hi : i < n) :
    let a := rankOneArguments input output n sx sy
    let q := i / a.B
    q % a.C < a.C ∧ q / a.C < a.T ∧ i % a.B < a.B ∧
      CappedGrid.logicalIndex a.B a.C (q % a.C) (q / a.C) (i % a.B) = i := by
  exact CappedGrid.logical_index_covered (block_pos (by omega)) hi

theorem prepared_branch (input output : Buffer) {n : Nat}
    (hn : 0 < n) (sx sy : Int) :
    (rankOneArguments input output n sx sy).oneTile = Bool.true ↔
      CappedGrid.numTiles n (block n) ≤ 65536 := by
  simpa [rankOneArguments] using CappedGrid.one_tile_iff hn (block_pos hn)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostPlan
