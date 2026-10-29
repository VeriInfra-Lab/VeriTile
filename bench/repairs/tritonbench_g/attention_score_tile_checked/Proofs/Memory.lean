import bench.repairs.tritonbench_g.attention_score_tile_checked.Contract
import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ByteMemory

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton Reused Geometry Physical FPStorage NativeInstructions
namespace Bytes
export VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory (Memory read words stores stores_render render_read render_frame)
end Bytes
noncomputable section

abbrev Binding (c : Shape) (abi : ABI c) (t : Tensors c) :=
  Reused.Contract.PointerBinding c abi (observations t)

def physicalReal (read : Nat → ℝ) : Nat → ℝ := fun word => read (4*word)

def logicalInputs (c : Shape) (t : Tensors c) (read : Nat → ℝ) : LogicalInputs ℝ c where
  q := fun z h i d => logical read t "q" (c.queryBase z.val h.val+i.val*c.D+d.val)
  k := fun z h j d => logical read t "k" ((z.val*c.HK+h.val)*c.NK*c.D+j.val*c.D+d.val)
  m := fun z h i => logical read t "m" (c.normalizerBase z.val h.val+i.val)

theorem key_all_heads_bound (c : Shape) (z : Fin c.Z) (h : Fin c.HK) (j : Fin c.NK) (d : Fin c.D) :
    (z.val*c.HK+h.val)*c.NK*c.D+j.val*c.D+d.val < c.keySize := by
  have a := digit_bound z.val c.Z h.val c.HK z.isLt h.isLt
  have b := digit_bound (z.val*c.HK+h.val) (c.Z*c.HK) j.val c.NK a j.isLt
  have e := digit_bound ((z.val*c.HK+h.val)*c.NK+j.val) (c.Z*c.HK*c.NK) d.val c.D b d.isLt
  simpa [Shape.keySize,Nat.add_mul,Nat.add_assoc] using e

theorem inputs_direct (c : Shape) (t : Tensors c) (abi : ABI c) (read : Nat → ℝ)
    (offset size : Int) (window : Bool) (admitted : InputContract c t offset size window)
    (launch : NonemptyLaunch c) (binding : Binding c abi t) :
    Normalized c abi (physicalReal read) (logicalInputs c t read) := by
  have direct (r : RegionName) (hr : tracked r) (i : Nat) (hi : i < Physical.size c r) :
      logical read t r i = read (4*((abi.alloc r).base+i)) := by
    rw [logical_direct read t admitted.1.1 r hr i hi,←binding launch.1 launch.2 r hr,Nat.mul_add]
  constructor
  · intro z h i d
    simpa [physicalReal,logicalInputs,Nat.add_assoc] using
      (direct "q" (by simp [tracked]) _ (by simpa [Physical.size,Nat.add_assoc] using query_bound c z h i d)).symm
  · intro z h j d
    simpa [physicalReal,logicalInputs,Nat.add_assoc] using
      (direct "k" (by simp [tracked]) _ (by simpa [Physical.size,Nat.add_assoc] using key_all_heads_bound c z h j d)).symm
  · intro z h i
    have hi : c.normalizerBase z.val h.val+i.val<c.normalizerSize :=
      digit_bound _ _ _ _ (digit_bound _ _ _ _ z.isLt h.isLt) i.isLt
    simpa [physicalReal,logicalInputs,Nat.add_assoc] using
      (direct "m" (by simp [tracked]) _ (by simpa [Physical.size] using hi)).symm

/-- Word-indexed source loads contain the exact bytes of the actual Tensor.
The `.real` carrier is the DSL's FP32-profile channel, not a Real conversion. -/
theorem input_words (c : Shape) (t : Tensors c) (abi : ABI c) (heap : Bytes.Memory)
    (offset size : Int) (window : Bool) (admitted : InputContract c t offset size window)
    (launch : NonemptyLaunch c) (binding : Binding c abi t)
    (r : RegionName) (hr : tracked r) (i : Nat) (hi : i<Physical.size c r) :
    (Physical.initial abi (NativePhysical.input (Bytes.words heap))).mem r i =
      .mk .real (.ready (.bits (Torch.FP32.logicalValue (fun a => ⟨Bytes.read heap a⟩)
        (t.get r) (t.api r hr) i).bits)) := by
  rw [logical_fp32 _ t admitted.1.1 r hr i hi,←binding launch.1 launch.2 r hr,←Nat.mul_add]
  rfl

/-- Mathematical cell-store projection, with no output formula built in. -/
def realMemory (c : Shape) (t : Tensors c) (result : Nat → ℝ) (before : Nat → ℝ) : Nat → ℝ :=
  fun address => if (observations t).pointer "out" ≤ address ∧
      address < (observations t).pointer "out"+4*c.outputSize ∧
      (address-(observations t).pointer "out")%4=0
    then result (address/4) else before address

/-- Native scalar/fixed-MMA operations are explicit model data. Loaded input
bits remain themselves; this is not a deferred numerical equality premise. -/
def nativeISA (I : ISA (BitVec 32)) : ISA (BitVec 32) := { I with bits := id }

def nativeMemory (I : ISA (BitVec 32)) (c : Shape) (t : Tensors c)
    (abi : ABI c) (heap : Bytes.Memory) (result : Nat → Graph) : Bytes.Memory :=
  Bytes.stores heap ((observations t).pointer "out") c.outputSize
    (fun i => (result ((abi.alloc "out").base+i)).eval (nativeISA I))

theorem real_read (c : Shape) (t : Tensors c) (abi : ABI c) (read : Nat → ℝ)
    (result : Nat → ℝ) (offset size : Int) (window : Bool)
    (admitted : InputContract c t offset size window) (launch : NonemptyLaunch c)
    (binding : Binding c abi t) (i : Nat) (hi : i<c.outputSize) :
    logical (realMemory c t result read) t "out" i = result ((abi.alloc "out").base+i) := by
  rw [logical_direct _ t admitted.1.1 "out" (by simp [tracked]) i (by simpa [Physical.size] using hi)]
  unfold realMemory
  have ptr := binding launch.1 launch.2 "out" (by simp [tracked])
  rw [←ptr]
  simp only [show 4*(abi.alloc "out").base ≤ 4*(abi.alloc "out").base+4*i ∧
    4*(abi.alloc "out").base+4*i < 4*(abi.alloc "out").base+4*c.outputSize ∧
    (4*(abi.alloc "out").base+4*i-4*(abi.alloc "out").base)%4=0 by omega,if_true]
  simp [←Nat.mul_add]

theorem native_read (I : ISA (BitVec 32)) (c : Shape) (t : Tensors c) (abi : ABI c)
    (heap : Bytes.Memory) (result : Nat → Graph) (offset size : Int) (window : Bool)
    (admitted : InputContract c t offset size window) (i : Nat) (hi : i<c.outputSize) :
    (Torch.FP32.logicalValue (fun a => ⟨Bytes.read (nativeMemory I c t abi heap result) a⟩)
      (t.get "out") (t.api "out" (by simp [tracked])) i).bits =
        (result ((abi.alloc "out").base+i)).eval (nativeISA I) := by
  rw [logical_fp32 _ t admitted.1.1 "out" (by simp [tracked]) i (by simpa [Physical.size] using hi)]
  change Bytes.read (nativeMemory I c t abi heap result) ((observations t).pointer "out"+4*i) = _
  rw [nativeMemory,Bytes.stores_render]
  exact Bytes.render_read _ _ _ _ i hi

def OutsideOutput (c : Shape) (t : Tensors c) (address : Nat) : Prop :=
  address < (observations t).pointer "out" ∨ (observations t).pointer "out"+4*c.outputSize ≤ address

theorem real_frame (c : Shape) (t : Tensors c) (result : Nat → ℝ) (read : Nat → ℝ)
    (address : Nat) (outside : OutsideOutput c t address) : realMemory c t result read address = read address := by
  unfold realMemory OutsideOutput at *
  simp [show ¬((observations t).pointer "out" ≤ address ∧
    address < (observations t).pointer "out"+4*c.outputSize ∧
    (address-(observations t).pointer "out")%4=0) by omega]

theorem native_frame (I : ISA (BitVec 32)) (c : Shape) (t : Tensors c) (abi : ABI c)
    (heap : Bytes.Memory) (result : Nat → Graph) (address : Nat) (outside : OutsideOutput c t address) :
    nativeMemory I c t abi heap result address = heap address := by
  unfold nativeMemory
  rw [Bytes.stores_render]
  exact Bytes.render_frame _ _ _ _ address outside

#axiomsClean inputs_direct
#axiomsClean input_words
#axiomsClean native_read
#axiomsClean native_frame
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
