import bench.repairs.tritonbench_g.masked_select.Kernels
import bench.repairs.tritonbench_g.masked_select.Proofs.Contract
import bench.repairs.tritonbench_g.masked_select.Proofs.Kernel
import VeriTile.Meta.StatementAudit

/-! Kernel-only stable compaction. The three prepared read buffers are kernel
inputs: FP32 payload, Bool selection, and inclusive Int64 prefix. Torch's
producer operations and wrapper execution are not proof obligations here. -/
namespace VeriTile.Bench.Repairs.MaskedSelect.KernelResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect
open TypedCopy Whole StableSelection
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section

/-- Exact selected output positions and an active-write frame for one full
kernel program. Stable ranks derive from the prefix input; no assumed output
or user-supplied no-collision fact appears in the contract. -/
theorem masked_select_kernel_correctness (rest : Algebra ℝ) (p : Parameters)
    (input : Nat → ℝ) (mask : Nat → Bool) (before : State ℝ)
    (h : KernelInputs p before input mask) :
    Spec.Real (∃ after, FP.Structural.exec (copyBackend rest) (kernel p) before = some after ∧
      (∀ lane : TileIndex [p.width], mask (inputOffset p before lane) = Bool.true →
        inputOffset p before lane < p.size →
        after.mem p.output (count mask (inputOffset p before lane)) =
          .mk .real (input (inputOffset p before lane))) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ lane : TileIndex [p.width],
        mask (inputOffset p before lane) = Bool.true → inputOffset p before lane < p.size →
        o ≠ count mask (inputOffset p before lane)) → after.mem r o = before.mem r o)) :=
  Kernel.program_correct rest p input mask before h

/-- The complete kernel configuration produces the independently defined
stable list, with length equal to the number of true mask entries, and frames
all cells outside that output window. Existing kernel composition is reused. -/
theorem masked_select_stable_correctness (rest : Algebra ℝ) (p : Parameters)
    (config : Config) (hc : config ∈ configs) (input : Nat → ℝ) (mask : Nat → Bool)
    (before : State ℝ) (h : KernelInputs p before input mask) (sep : KernelSeparate p) :
    Spec.Real (∃ after, Kernel.run rest p config before = some after ∧
      Kernel.Result p input mask before after) :=
  Kernel.complete rest p config hc input mask before h sep

/-- Inclusive Int64 prefixes and selected destinations are exact. A native
zero-prefix `-1` occurs only where the store is masked off. -/
theorem prefix_int64_meaning (mask : Nat → Bool) (n i : Nat) (hn : n < 2^31) (hi : i < n) :
    (prefix64 mask i).toInt = (count mask (i + 1) : Nat) ∧
    (mask i = Bool.true → (destination64 mask i).toInt = (count mask i : Nat) ∧
      count mask i < count mask n) ∧
    (count mask (i + 1) = 0 → mask i = Bool.false ∧ (destination64 mask i).toInt = -1) :=
  ⟨prefix64_exact mask n i hn hi,destination64_selected mask n i hn hi,zero_prefix_masked mask i⟩

#specNonCircular Kernel.mathematical avoiding [kernel,Kernel.run]
#axiomsClean masked_select_kernel_correctness
#axiomsClean masked_select_stable_correctness
#axiomsClean prefix_int64_meaning
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.KernelResults
