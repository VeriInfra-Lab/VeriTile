import bench.tritonbench_g.f8_conversion_utils.Proofs.Memory
import bench.tritonbench_g.f8_conversion_utils.Proofs.ByteStore

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Bytes
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
open ByteStore
noncomputable section
abbrev Heap := Nat → BitVec 8
abbrev Word (width : Nat) := Fin width → BitVec 8

def read (width : Nat) (memory : Heap) (base : Nat) : Word width :=
  fun byte => memory (base+byte.val)

/-- A concrete mixed-width byte machine. Only output-sized words can be
stored; no roundtrip for arbitrary Real or differently sized input values is
postulated. The imported word writer performs every little-endian byte update. -/
def primitives (iw ow : Nat) : Memory.Primitives Heap (Word iw) (Word ow) iw ow where
  readInput := read iw
  readOutput := read ow
  writeOutput := fun s a value => writeWord a value s
  byte := fun s a => (s a).toFin
  read_same := by
    intro s a value
    funext k
    exact writeWord_value a value s k
  read_disjoint := by
    intro s a value b apart
    funext k
    apply writeWord_frame
    intro j same
    rcases apart with h|h <;> have hk := k.isLt <;> have hj := j.isLt <;> omega
  input_disjoint := by
    intro s a value b apart
    funext k
    apply writeWord_frame
    intro j same
    rcases apart with h|h <;> have hk := k.isLt <;> have hj := j.isLt <;> omega
  byte_frame := by
    intro s a value b outside
    apply congrArg BitVec.toFin
    apply writeWord_frame
    intro k same
    have hk := k.isLt
    rcases outside with h|h <;> omega

theorem output_at (iw ow : Nat) (s : Heap) (a : Nat) (v : Word ow)
    (equal : (primitives iw ow).readOutput s a=v) (k : Fin ow) : s (a+k.val)=v k :=
  congrFun equal k

theorem byte_eq (iw ow : Nat) (s t : Heap) (a : Nat)
    (equal : (primitives iw ow).byte s a=(primitives iw ow).byte t a) : s a=t a :=
  BitVec.eq_of_toFin_eq equal

/-- Readback is a theorem about the same executed raw memory, for every output
byte. It is not a post-hoc replacement of memory by expected results. -/
theorem complete (iw ow : Nat) (twice : Bool) (convert : Word iw → Word ow)
    (input output n : Nat) (before final : Heap)
    (executed : final=Memory.run (primitives iw ow) twice convert input output (Chunking.batches n) before)
    (apart : ∀ i,i<n → ∀ j,j<n →
      SpanDisjoint (Memory.address output ow i) (Memory.address output ow i+ow)
        (Memory.address input iw j) (Memory.address input iw j+iw)) :
    (∀ i,i<n → ∀ k : Fin ow,
      final (Memory.address output ow i+k.val)=convert (read iw before (Memory.address input iw i)) k) ∧
    (∀ q,(∀ i,i<n → q<Memory.address output ow i ∨ Memory.address output ow i+ow≤q) →
      final q=before q) := by
  have checked := Memory.complete (primitives iw ow) twice convert input output n
    (Chunking.batches n) before (Chunking.full_coverage n) apart
  rw [←executed] at checked
  constructor
  · intro i hi k
    exact output_at iw ow final _ _ (checked.1 i hi) k
  · intro q outside
    have equal := checked.2 q outside
    exact byte_eq iw ow final before q equal

#axiomsClean primitives
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Bytes
