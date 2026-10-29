import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.Indices

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelInputs
open VeriTile.Triton FP.Structural KernelModels KernelResult
open QuantTransposeFinite (Input Allocations PhysicalLayout)
noncomputable section

def parameters := QuantTransposeFinite.parameters

/-- Typed initial reads connect the actual shared kernel to the existing
Tensor/storage model. Only source inputs occur here. Inverse bytes describe
the actual argument at launch; no host computation is proved or assumed to
produce a desired output. -/
structure NativeReads (a : Input) (alloc : Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String) (physical : PhysicalLayout)
    (inverseValue : Word) (s : State Word) : Prop where
  input : ∀ o w, memory x o = some (.mk .fp32 w) →
    s.mem ⟨x⟩ o = .mk .real (FP.Value.mk w)
  inverse : s.mem ⟨alloc.scale⟩ 0 = .mk .real inverseValue
  inverseBytes : ∀ k, k < 4 → physical.initialBytes (physical.inverse.base + k) =
    some (QuantTransposeFinite.byteOf (d := .fp32) inverseValue.bits k)

theorem separated (a : Input) (alloc : Allocations) (x : String) (fresh : alloc.Fresh x) :
    KernelGrid.Separate (parameters a alloc x) :=
  ⟨fresh.2.2.1, fresh.2.2.2.2.2⟩

theorem native_reads (a : Input) (alloc : Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32) (physical : PhysicalLayout)
    (inverse : Word) (s : State Word)
    (domain : QuantTransposeFinite.Domain .fp32 a alloc memory x logical xs)
    (reads : NativeReads a alloc memory x physical inverse s) :
    KernelWhole.Reads (parameters a alloc x)
      (fun r c => (⟨xs (c * a.M + r)⟩ : Word)) inverse s := by
  constructor
  · intro r c hr hc
    have hb := QuantTransposeFinite.active_encode_bound a.M a.N r c hr hc
    have hd := QuantTransposeFinite.encode_decode a.M r c hr
    have hm := domain.strided (c * a.M + r) hb
    have hp : QuantTransposeFinite.inputOffset (parameters a alloc x) (c * a.M + r) =
        r * a.sam + c * a.san := by
      simp [parameters, QuantTransposeFinite.parameters, QuantTransposeFinite.inputOffset, hd.1, hd.2]
    rw [show QuantTransposeFinite.inputOffset (QuantTransposeFinite.parameters a alloc x)
      (c * a.M + r) = r * a.sam + c * a.san from hp] at hm
    exact reads.input _ _ hm
  · exact reads.inverse

theorem input_bytes (a : Input) (alloc : Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32) (physical : PhysicalLayout)
    (domain : QuantTransposeFinite.Domain .fp32 a alloc memory x logical xs)
    (binding : QuantTransposeFinite.PhysicalBinding .fp32 a alloc memory x physical)
    (r c k : Nat) (hr : r < a.M) (hc : c < a.N) (hk : k < 4) :
    physical.initialBytes (physical.inputPointer.toNat + (r * a.sam + c * a.san) * 4 + k) =
      some (QuantTransposeFinite.byteOf (d := .fp32) (xs (c * a.M + r)) k) := by
  have hb := QuantTransposeFinite.active_encode_bound a.M a.N r c hr hc
  have hd := QuantTransposeFinite.encode_decode a.M r c hr
  have hm := domain.strided (c * a.M + r) hb
  have h := binding.inputWords
    (QuantTransposeFinite.inputOffset (parameters a alloc x) (c * a.M + r))
    (Or.inr ⟨c * a.M + r, by simpa [domain.matrixCount] using hb, rfl⟩)
    (xs (c * a.M + r)) hm k hk
  simpa [parameters, QuantTransposeFinite.parameters, QuantTransposeFinite.inputOffset,
    QuantTransposeFinite.bytes, QuantTransposeFinite.Storage.width, hd.1, hd.2] using h

/-- The existing checked signed-int32 arithmetic proof now names the exact
lane offsets used by the full shared source. Native pointer arithmetic and
the transposed Tensor result therefore use the same addresses. -/
theorem native_addresses (md : Metadata) (admitted : InputContract md)
    (alloc : Allocations) (memory : QuantTransposeFinite.Memory) (x : String)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32) (physical : PhysicalLayout)
    (domain : QuantTransposeFinite.Domain .fp32 md.input alloc memory x logical xs)
    (ordinary : OrdinaryPhysicalBinding md.input alloc memory x physical)
    (pointer : physical.inputPointer.toNat = md.dataPointer)
    (pid : Nat) (hp : pid < QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N)
    (i : Lane) :
    ∃ native, QuantTransposeFinite.nativeLane md.input.M md.input.N md.input.sam md.input.san
      pid (flat i) = some native ∧
      native.mask = decide (active (parameters md.input alloc x) pid i) ∧
      (active (parameters md.input alloc x) pid i →
        (QuantTransposeFinite.nativePointer physical.inputPointer.toNat 4 native.inputOffset).toNat =
          physical.inputPointer.toNat + inputOffset (parameters md.input alloc x) pid i * 4 ∧
        (QuantTransposeFinite.nativePointer physical.output.base 1 native.outputOffset).toNat =
          physical.output.base + outputOffset (parameters md.input alloc x) pid i) := by
  let binding := ordinary.admitted admitted pointer
  have refined := arithmetic_refines md.input admitted.indices .fp32 alloc memory x logical xs
    physical domain binding
  obtain ⟨native, hn, hm, ha⟩ := refined pid (flat i) hp (flat_bound i)
  refine ⟨native, hn, ?_, ?_⟩
  · simpa [active, ← flat_row, ← flat_col, parameters, QuantTransposeFinite.parameters] using hm
  · intro hi
    have hm' : native.mask = Bool.true := by
      rw [hm]
      apply decide_eq_true
      simpa [QuantTransposeFinite.active, parameters, QuantTransposeFinite.parameters] using
        (flat_active (parameters md.input alloc x) pid i).2 hi
    simpa [parameters, QuantTransposeFinite.parameters, QuantTransposeFinite.bytes,
      QuantTransposeFinite.Storage.width, inputOffset, outputOffset, ← flat_row, ← flat_col] using ha hm'

#axiomsClean native_reads
#axiomsClean input_bytes
#axiomsClean native_addresses
#specNonCircular NativeReads avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelInputs
