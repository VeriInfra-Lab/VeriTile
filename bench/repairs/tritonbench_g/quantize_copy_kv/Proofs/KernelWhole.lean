import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelGridResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelInputs
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelWhole
open VeriTile.Triton FP.Structural KernelSource KernelInputs
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 2500000

/-- All logical Tensor outputs and the complete exterior memory frame. The
target functions are independent of execution, registers, and store events. -/
def TensorResult {α : Type} (m : Inputs) (words : Nat → MetadataWord)
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (initial final : State α) : Prop :=
  (∀ token head column, token < (request m words).tokens → head < (request m words).heads →
    column < (request m words).columns →
    final.mem "output" (KernelResult.codeOffset (parameters m words) ((request m words).destination token) head column) =
      .mk .int (codes token head column)) ∧
  (∀ token head, token < (request m words).tokens → head < (request m words).heads →
    final.mem "scale" (KernelResult.scaleOffset (parameters m words) ((request m words).destination token) head) =
      .mk .fp16 (scales token head)) ∧
  (∀ (r : RegionName) o,
    (r ≠ "output" ∨ ∀ token head column, token < (request m words).tokens →
      head < (request m words).heads → column < (request m words).columns →
      o ≠ KernelResult.codeOffset (parameters m words) ((request m words).destination token) head column) →
    (r ≠ "scale" ∨ ∀ token head, token < (request m words).tokens → head < (request m words).heads →
      o ≠ KernelResult.scaleOffset (parameters m words) ((request m words).destination token) head) →
    final.mem r o = initial.mem r o)

theorem whole {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (initial : State α)
    (metadata : ∀ token, token < (request m words).tokens →
      (initial.mem "destination" token).read .int = ((request m words).destination token : Int))
    (codes : Nat → Nat → Nat → Int) (scales : Nat → Nat → α)
    (code_law : ∀ token (i : TileIndex [(parameters m words).height, (parameters m words).width]),
      token < (request m words).tokens → i.1.val < (request m words).heads →
      KernelResult.code M convert equal (parameters m words) (KernelGrid.atProgram initial token) i =
        codes token i.1.val i.2.1.val)
    (scale_law : ∀ token (head : Fin (parameters m words).height),
      token < (request m words).tokens → head.val < (request m words).heads →
      KernelResult.scale M (parameters m words) (KernelGrid.atProgram initial token) head = scales token head.val) :
    ∃ final,
      KernelGrid.run (KernelExecution.backend M convert equal) (parameters m words) Bool.false
        (List.range (request m words).tokens) initial = some final ∧
      KernelGrid.run (KernelExecution.backend M convert equal) (parameters m words) Bool.true
        (List.range (request m words).tokens) initial = some final ∧
      TensorResult m words codes scales initial final := by
  let p := parameters m words
  let order := List.range (request m words).tokens
  have hw := width_positive m words h
  let final := KernelGridResult.finalState M convert equal p hw order initial
  have run := KernelGridResult.run_finalState M convert equal p hw order initial
  refine ⟨final, run, (KernelGrid.same_run _ p order initial).symm.trans run, ?_⟩
  refine ⟨?_, ?_, ?_⟩
  · intro token head column ht hh hc
    let i : TileIndex [p.height, p.width] :=
      (⟨head, lt_of_lt_of_le hh (heads_covered m words)⟩, ⟨column, hc⟩, PUnit.unit)
    have result := KernelGridResult.final_code M convert equal p hw initial _ _ metadata h.unique
      order (fun _ hp => List.mem_range.mp hp) List.nodup_range token (List.mem_range.mpr ht) i hh
    rw [code_law token i ht hh] at result
    exact result
  · intro token head ht hh
    let i : TileIndex [p.height, 1] :=
      (⟨head, lt_of_lt_of_le hh (heads_covered m words)⟩, ⟨0, by decide⟩, PUnit.unit)
    have result := KernelGridResult.final_scale M convert equal p hw initial _ _ metadata h.unique
      order (fun _ hp => List.mem_range.mp hp) List.nodup_range token (List.mem_range.mpr ht) i hh
    rw [scale_law token i.1 ht hh] at result
    exact result
  · intro r o codes_miss scales_miss
    exact KernelGridResult.final_frame M convert equal p hw initial _ _ metadata order
      (fun _ hp => List.mem_range.mp hp) r o codes_miss scales_miss initial ⟨fun _ => rfl, fun _ => rfl⟩

theorem mathematical (rest : Algebra ℝ) (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (reads : RealReads m words values initial) :
    ∃ final,
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) (parameters m words) Bool.false
        (List.range (request m words).tokens) initial = some final ∧
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) (parameters m words) Bool.true
        (List.range (request m words).tokens) initial = some final ∧
      TensorResult m words
        (fun token head column => KernelRealModel.code (values token head) (request m words).columns column)
        (fun token head => KernelRealModel.scale (values token head) (request m words).columns) initial final := by
  apply whole (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal m words h initial
    (real_destination m words h values initial reads)
  · intro token i ht hh
    apply KernelRealResult.code rest _ _ (width_positive m words h) i.1 hh (values token i.1.val)
    intro c hc
    simpa [KernelGrid.atProgram, KernelResult.inputOffset, parameters, inputCell] using
      reads.input token i.1.val c ht hh hc
  · intro token head ht hh
    apply KernelRealResult.scale rest _ _ (width_positive m words h) head hh (values token head.val)
    intro c hc
    simpa [KernelGrid.atProgram, KernelResult.inputOffset, parameters, inputCell] using
      reads.input token head.val c ht hh hc

def tensorWords (m : Inputs) (words : Nat → MetadataWord) (heap : Heap)
    (token head column : Nat) : Word :=
  decodeWord (fun byte => heap (base m .input + 4 * inputCell (request m words) token head column + byte.val))

def nativeCode (a : KernelNativeProfile.Arithmetic) (m : Inputs) (words : Nat → MetadataWord) (heap : Heap)
    (token head column : Nat) : Int :=
  if hh : head < (parameters m words).height then
    if hc : column < (parameters m words).width then
      KernelNativeResult.code a (parameters m words) (tensorWords m words heap token)
        (⟨head, hh⟩, ⟨column, hc⟩, PUnit.unit)
    else 0
  else 0

def nativeScale (a : KernelNativeProfile.Arithmetic) (m : Inputs) (words : Nat → MetadataWord) (heap : Heap)
    (token head : Nat) : Word :=
  if hh : head < (parameters m words).height then
    MixedBits.embed (KernelNativeResult.scale a (parameters m words) (tensorWords m words heap token) ⟨head, hh⟩)
  else FP.Value.zero .fp32

theorem native (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words)
    (heap : Heap) (binding : TensorBinding m words heap) (initial : State Word)
    (reads : NativeReads m words heap initial) :
    ∃ final,
      KernelGrid.run (KernelNativeProfile.backend a rest) (parameters m words) Bool.false
        (List.range (request m words).tokens) initial = some final ∧
      KernelGrid.run (KernelNativeProfile.backend a rest) (parameters m words) Bool.true
        (List.range (request m words).tokens) initial = some final ∧
      TensorResult m words (nativeCode a m words heap) (nativeScale a m words heap) initial final := by
  apply whole (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal m words h initial
    (native_destination m words heap initial binding reads)
  · intro token i ht hh
    simp only [nativeCode, i.1.isLt, i.2.1.isLt, dite_true, Fin.eta]
    apply KernelNativeResult.code_eq
    intro head column ha hc
    simpa [KernelGrid.atProgram, KernelResult.inputOffset, parameters, inputCell, tensorWords] using
      reads.input token head column ht ha hc
  · intro token head ht hh
    simp only [nativeScale, head.isLt, dite_true, Fin.eta]
    apply KernelNativeResult.scale_eq
    intro a c ha hc
    simpa [KernelGrid.atProgram, KernelResult.inputOffset, parameters, inputCell, tensorWords] using
      reads.input token a c ht ha hc

#axiomsClean whole
#axiomsClean mathematical
#axiomsClean native
#specNonCircular TensorResult avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
#specNonCircular nativeCode avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
#specNonCircular nativeScale avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelWhole
