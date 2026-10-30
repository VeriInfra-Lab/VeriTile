import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelGridResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.RawKernel
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 2500000

/-- All logical kernel outputs and the complete exterior memory frame. The
target functions are independent of execution, registers, and store events. -/
def Result {α : Type} (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (initial final : State α) : Prop :=
  (∀ token head column, token < count → head < p.heads →
    column < p.width →
    final.mem "output" (KernelResult.codeOffset p (dest token) head column) =
      .mk .int (codes token head column)) ∧
  (∀ token head, token < count → head < p.heads →
    final.mem "scale" (KernelResult.scaleOffset p (dest token) head) =
      .mk .fp16 (scales token head)) ∧
  (∀ (r : RegionName) o,
    (r ≠ "output" ∨ ∀ token head column, token < count →
      head < p.heads → column < p.width →
      o ≠ KernelResult.codeOffset p (dest token) head column) →
    (r ≠ "scale" ∨ ∀ token head, token < count → head < p.heads →
      o ≠ KernelResult.scaleOffset p (dest token) head) →
    final.mem r o = initial.mem r o)

theorem whole {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (hw : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) (initial : State α)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (code_law : ∀ token (i : TileIndex [p.height, p.width]),
      token < count → i.1.val < p.heads →
      KernelResult.code M convert equal p (KernelGrid.atProgram initial token) i =
        codes token i.1.val i.2.1.val)
    (scale_law : ∀ token (head : Fin p.height),
      token < count → head.val < p.heads →
      KernelResult.scale M p (KernelGrid.atProgram initial token) head = scales token head.val) :
    ∃ final,
      KernelGrid.run (KernelExecution.backend M convert equal) p Bool.false
        (List.range count) initial = some final ∧
      KernelGrid.run (KernelExecution.backend M convert equal) p Bool.true
        (List.range count) initial = some final ∧
      Result p count dest codes scales initial final := by
  let order := List.range count
  let final := KernelGridResult.finalState M convert equal p hw order initial
  have run := KernelGridResult.run_finalState M convert equal p hw order initial
  refine ⟨final, run, (KernelGrid.same_run _ p order initial).symm.trans run, ?_⟩
  refine ⟨?_, ?_, ?_⟩
  · intro token head column ht hh hc
    let i : TileIndex [p.height, p.width] :=
      (⟨head, lt_of_lt_of_le hh covered⟩, ⟨column, hc⟩, PUnit.unit)
    have result := KernelGridResult.final_code M convert equal p hw initial _ _ metadata unique
      order (fun _ hp => List.mem_range.mp hp) List.nodup_range token (List.mem_range.mpr ht) i hh
    rw [code_law token i ht hh] at result
    exact result
  · intro token head ht hh
    let i : TileIndex [p.height, 1] :=
      (⟨head, lt_of_lt_of_le hh covered⟩, ⟨0, by decide⟩, PUnit.unit)
    have result := KernelGridResult.final_scale M convert equal p hw initial _ _ metadata unique
      order (fun _ hp => List.mem_range.mp hp) List.nodup_range token (List.mem_range.mpr ht) i hh
    rw [scale_law token i.1 ht hh] at result
    exact result
  · intro r o codes_miss scales_miss
    exact KernelGridResult.final_frame M convert equal p hw initial _ _ metadata order
      (fun _ hp => List.mem_range.mp hp) r o codes_miss scales_miss initial ⟨fun _ => rfl, fun _ => rfl⟩

def nativeCode (a : KernelNativeProfile.Arithmetic) (p : Parameters)
    (values : Nat → Nat → Nat → Word) (token head column : Nat) : Int :=
  if hh : head < p.height then
    if hc : column < p.width then
      KernelNativeResult.code a p (values token) (⟨head, hh⟩, ⟨column, hc⟩, PUnit.unit)
    else 0
  else 0

def nativeScale (a : KernelNativeProfile.Arithmetic) (p : Parameters)
    (values : Nat → Nat → Nat → Word) (token head : Nat) : Word :=
  if hh : head < p.height then
    MixedBits.embed (KernelNativeResult.scale a p (values token) ⟨head, hh⟩)
  else FP.Value.zero .fp32

#axiomsClean whole
#specNonCircular Result avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
#specNonCircular nativeCode avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
#specNonCircular nativeScale avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.RawKernel
