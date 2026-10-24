import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Contract
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelOffsets
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.IntegerEncoding
import VeriTile.Torch.FP32
import VeriTile.Triton.Memory.Bounds

namespace VeriTile.Bench.Repairs.UniformSampling.KernelMemory
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.UniformSampling
open KernelOffsets (output)

/-- Each wrapper chunk has at most 2^34 elements and uses an existing block
size. The local counter and all four output offsets fit their source widths. -/
theorem native_bounds (N pid B : Nat) (block : B=512 ∨ B=1024)
    (extent : N≤2^34) (launched : pid<SourceDomain.grid N B) (f : Fin 4) (i : Fin B) :
    pid<2^31 ∧ pid*B+i.val<2^32 ∧ output pid B f.val i.val<2^34 := by
  have hf := f.isLt
  have hi := i.isLt
  rcases block with rfl | rfl <;>
    norm_num [SourceDomain.grid,output] at launched ⊢ <;> omega

theorem native_exact (N pid B : Nat) (block : B=512 ∨ B=1024)
    (extent : N≤2^34) (launched : pid<SourceDomain.grid N B) (f : Fin 4) (i : Fin B) :
    (Bounds.counterBits pid B i.val).toInt = ((pid*B+i.val : Nat) : Int) ∧
    (Bounds.outputBits pid B i.val f.val).toInt = ((output pid B f.val i.val : Nat) : Int) := by
  have h := native_bounds N pid B block extent launched f i
  rw [Bounds.counter_bits,Bounds.output_bits]
  exact ⟨Bounds.i64_exact (by omega),Bounds.i64_exact (by simpa [output] using (by omega : output pid B f.val i.val<2^63))⟩

/-- Every eager clamped pointer stays in the typed output region, including
masked lanes. Active lanes retain their original address and mask. -/
theorem addresses (y : RegionName) (N pid B : Nat) (capacity : RegionBounds)
    (nonempty : 0<N) (bounds : N≤capacity y) (f : Fin 4) (i : Fin B) :
    min (output pid B f.val i.val) (N-1) < N ∧
    min (output pid B f.val i.val) (N-1) < capacity y ∧
    (output pid B f.val i.val<N →
      min (output pid B f.val i.val) (N-1)=output pid B f.val i.val ∧
      output pid B f.val i.val<capacity y) := by
  have hmin := Nat.min_le_right (output pid B f.val i.val) (N-1)
  refine ⟨by omega,by omega,?_⟩
  intro active
  exact ⟨Nat.min_eq_left (by omega),by omega⟩

/-- The eager clamp's natural subtraction agrees with the native signed
scalar expression precisely on the nonempty launch domain. -/
theorem clamp_bound (N : Nat) (nonempty : 0<N) (extent : N≤2^34) :
    (BitVec.ofNat 64 (N-1)).toInt = (N : Int)-1 := by
  rw [Bounds.i64_exact (by omega)]
  simp [Nat.cast_sub (by omega : 1≤N)]

/-- The existing Tensor guards bind contiguous FP32 storage to the shared
actual-pointer and logical-value interface. No allocation behavior is required. -/
theorem checked_output_memory (t : FP32.Tensor) (laws : FP32.TensorLaws t)
    (generator : GeneratorArgument) (lowerIsFloat upperIsFloat : Bool)
    (admitted : InputContract t.metadata generator lowerIsFloat upperIsFloat)
    (read : Nat → FP32.Word) :
    let view := FP32.dataView t laws admitted.contiguous
    view.base % 4 = 0 ∧
    (0<t.metadata.numel → BitVec.ofNat 64 (view.address 0) = t.dataPtr) ∧
    (∀ i,i<t.metadata.numel → read (view.address i)=FP32.logicalValue read t laws i) := by
  exact ⟨FP32.dataView_aligned t laws admitted.contiguous admitted.alignment,
    FP32.actual_pointer t laws admitted.contiguous,
    FP32.direct_read_eq_logical read t laws admitted.contiguous admitted.noNeg⟩

#axiomsClean native_bounds
#axiomsClean native_exact
#axiomsClean addresses
#axiomsClean clamp_bound
#axiomsClean checked_output_memory
end VeriTile.Bench.Repairs.UniformSampling.KernelMemory
