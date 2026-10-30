import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Memory
import bench.tritonbench_g.isfinite_kernel.Proofs.BitClassification

namespace VeriTile.Bench.Repairs.IsfiniteKernel
noncomputable section
open Classical

namespace Bytes
abbrev Heap := Nat → BitVec 8
abbrev Word (width : Nat) := Fin width → BitVec 8
def read (width : Nat) (s : Heap) (q : Nat) : Word width := fun k => s (q+k.val)
def write (s : Heap) (q : Nat) (value : Bool) : Heap := Function.update s q (Memory.boolByte value)

def primitives (width : Nat) : Memory.Primitives Heap (Word width) width where
  read := read width
  byte := id
  write := write
  write_byte := by intro s a b q; simp [write,Function.update_apply]
  input_frame := by
    intro s a b q separate
    funext k
    exact Function.update_of_ne (separate k.val k.isLt) _ _

def wordNat (word : Word width) : Nat := ((List.finRange width).map (fun k => (word k).toNat*256^k.val)).sum

/-- Independent IEEE reference: exponent all-ones rejects both infinities and
every NaN payload; every other exponent accepts, including signed zero and
all subnormals. FP64 classification never passes through FP32. -/
def reference (precision : Precision) (word : Word precision.width) : Bool :=
  match precision with
  | .single => VeriTile.Bench.TritonBenchG.IsfiniteKernel.BitClassification.finite32
      ⟨BitVec.ofNat 32 (wordNat word)⟩
  | .double => VeriTile.Bench.TritonBenchG.IsfiniteKernel.BitClassification.finite64
      ⟨BitVec.ofNat 64 (wordNat word)⟩

/-- Optional primitive-model interface for the retained host theorem.
It specifies finitef/isfinited on one physical word; it is not a numerical
transformation or an obligation to run a per-task numerical experiment. -/
def LocalClassifier (precision : Precision) (native : Word precision.width → Bool) : Prop :=
  ∀ word,native word=reference precision word
end Bytes

namespace RealStorage
/-- A mathematical Real cell is not encoded into a finite-width bitvector.
Byte sentinels and Real loads are separate observations of the ideal memory. -/
structure Heap where
  reals : Nat → ℝ
  bytes : Nat → BitVec 8

def write (s : Heap) (q : Nat) (value : Bool) : Heap :=
  ⟨Function.update s.reals q 0,Function.update s.bytes q (Memory.boolByte value)⟩

def primitives (width : Nat) (positive : 0<width) : Memory.Primitives Heap ℝ width where
  read := fun s q => s.reals q
  byte := fun s q => s.bytes q
  write := write
  write_byte := by intro s a b q; simp [write,Function.update_apply]
  input_frame := by
    intro s a b q separate
    exact Function.update_of_ne (by simpa only [Nat.add_zero] using separate 0 positive) _ _

def reference (_ : ℝ) : Bool := Bool.true
end RealStorage

#axiomsClean Bytes.primitives
#axiomsClean RealStorage.primitives
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel
