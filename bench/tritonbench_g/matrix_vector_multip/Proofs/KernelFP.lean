import bench.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_WholeNative

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelFP
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeSource WholeNative
noncomputable section
set_option maxHeartbeats 3000000

/-- Independent native instruction step on logical input words. Masked slots
still execute their FMA, including zero padding and the prior accumulator. -/
def next (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (pid off : Nat) (acc : Acc (BitVec 32) (parameters N M BN BM)) :
    Acc (BitVec 32) (parameters N M BN BM) := fun i =>
  let a := if pid * BN + i.1.val < N ∧ off + i.2.1.val < M then
    matrix (pid * BN + i.1.val) (off + i.2.1.val) else 0
  let b := if off + i.2.1.val < M then vector (off + i.2.1.val) else 0
  if g.reverseFMA i.1 i.2.1 then native.fma b a (acc i) else native.fma a b (acc i)

def accumulate (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (pid off : Nat) (acc : Acc (BitVec 32) (parameters N M BN BM)) :
    Acc (BitVec 32) (parameters N M BN BM) :=
  if h : off < M ∧ 0 < BM then
    accumulate native N M BN BM g matrix vector pid (off + BM)
      (next native N M BN BM g matrix vector pid off acc)
  else acc
termination_by M - off

/-- Complete ordered FP32 result graph, independently evaluated without the
source executor. Tree completeness records coverage, never associativity. -/
def formula (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (pid : Nat) (i : Fin BN) : BitVec 32 :=
  (g.tree i).eval native.add (fun j =>
    accumulate native N M BN BM g matrix vector pid 0 (fun _ => 0) (i, j, PUnit.unit))

/-- Only active input words are pinned. Other typed memory cells, including
masked addresses, are arbitrary and are preserved by the frame theorem. -/
structure InputWords (N M : Nat) (s : State Payload)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32) : Prop where
  matrix : ∀ i j, i < N → j < M →
    (s.mem "A" (i * M + j)).read .fp32 = .word (matrix i j)
  vector : ∀ j, j < M → (s.mem "B" j).read .fp32 = .word (vector j)

variable {N M BN BM : Nat} {s : State Payload}
    {matrix : Nat → Nat → BitVec 32} {vector : Nat → BitVec 32}

theorem matrix_chunk (native : Scalars) (g : Graph BN BM) (h : InputWords N M s matrix vector)
    (pid off : Nat) (i : TileIndex [BN, BM]) :
    aChunk (algebra native g) (parameters N M BN BM) pid off s.mem i =
      .word (if pid * BN + i.1.val < N ∧ off + i.2.1.val < M then
        matrix (pid * BN + i.1.val) (off + i.2.1.val) else 0) := by
  simp only [aChunk, load, algebra, parameters, WholeSource.row, aOffset, Nat.mul_one]
  split
  · rename_i active
    have address : (pid * BN + i.1.val) * M + i.2.1.val + off =
        (pid * BN + i.1.val) * M + (off + i.2.1.val) := by omega
    rw [address, h.matrix _ _ active.1 active.2]
  · rfl

theorem vector_chunk (native : Scalars) (g : Graph BN BM) (h : InputWords N M s matrix vector)
    (off : Nat) (i : TileIndex [1, BM]) :
    bChunk (algebra native g) (parameters N M BN BM) off s.mem i =
      .word (if off + i.2.1.val < M then vector (off + i.2.1.val) else 0) := by
  simp only [bChunk, load, algebra, parameters, bOffset, Nat.mul_one]
  split
  · rename_i active
    rw [Nat.add_comm i.2.1.val off, h.vector _ active]
  · rfl

theorem next_eq (native : Scalars) (g : Graph BN BM) (h : InputWords N M s matrix vector)
    (pid off : Nat) (acc : Acc Payload (parameters N M BN BM)) :
    evaluateAcc native (parameters N M BN BM) g
      (nextAcc (algebra native g) (parameters N M BN BM) pid off s.mem acc) =
    next native N M BN BM g matrix vector pid off
      (evaluateAcc native (parameters N M BN BM) g acc) := by
  funext i
  simp only [evaluateAcc, nextAcc]
  rw [matrix_chunk native g h, vector_chunk native g h]
  rfl

theorem accumulated_eq (native : Scalars) (g : Graph BN BM) (h : InputWords N M s matrix vector)
    (pid off : Nat) (acc : Acc Payload (parameters N M BN BM)) :
    evaluateAcc native (parameters N M BN BM) g
      (accumulated (algebra native g) (parameters N M BN BM) pid s.mem off acc) =
    accumulate native N M BN BM g matrix vector pid off
      (evaluateAcc native (parameters N M BN BM) g acc) := by
  conv_lhs => rw [accumulated]
  conv_rhs => rw [accumulate]
  simp only [show (parameters N M BN BM).columns = M from rfl,
    show (parameters N M BN BM).blockColumns = BM from rfl]
  split
  · rw [accumulated_eq native g h, next_eq native g h]
  · rfl
termination_by M - off

theorem formula_eq (native : Scalars) (g : Graph BN BM) (h : InputWords N M s matrix vector)
    (pid : Nat) (i : Fin BN) :
    tileValue (algebra native g) (parameters N M BN BM) pid s.mem i =
      .word (formula native N M BN BM g matrix vector pid i) := by
  unfold tileValue storedValue outputValues
  change sum native g ⟨1, by simp⟩ Bool.false
    (accumulated (algebra native g) (parameters N M BN BM) pid s.mem 0 (fun _ => .word 0))
      (i, PUnit.unit) = _
  simp only [sum, show (parameters N M BN BM).blockRows = BN from rfl,
    show (parameters N M BN BM).blockColumns = BM from rfl, dif_pos rfl]
  unfold sumTile
  change Payload.word ((g.tree i).eval native.add (fun j =>
    evaluateAcc native (parameters N M BN BM) g
      (accumulated (algebra native g) (parameters N M BN BM) pid s.mem 0 (fun _ => .word 0))
      (i, j, PUnit.unit))) = _
  rw [accumulated_eq native g h]
  rfl

#axiomsClean formula_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelFP
