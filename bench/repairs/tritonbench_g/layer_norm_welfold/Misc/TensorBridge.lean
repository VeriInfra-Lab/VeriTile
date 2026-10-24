import bench.repairs.tritonbench_g.layer_norm_welfold.Correct
import bench.repairs.tritonbench_g.layer_norm_welfold.FPEquiv
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeInputs
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Physical

/-! Full Tensor applications of the raw mathematical and native kernel interfaces. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.TensorBridge
open VeriTile.Triton FP.Structural Reuse Samples
open VeriTile.Torch.FP32 (Word TensorLaws)
noncomputable section

theorem python_checks_iff_preconditions (W Bias X : Admission.Argument) (S D : Nat) :
    Admission.checkInputs W Bias X = .ok (S,D) ↔ Admission.Admitted W Bias X S D :=
  Admission.checkInputs_iff W Bias X S D

section Real
open InputBinding

/-- Actual source success, all mean/rstd/BF16-Y cells, the full exterior frame,
and original logical/physical input meaning in one composed theorem. -/
theorem layer_norm_welfold_real (rest : Algebra ℝ) (read : Nat → Word) (decode : Word → ℝ)
    (x w bias : VeriTile.Torch.FP32.Tensor) (lx : TensorLaws x) (lw : TensorLaws w) (lb : TensorLaws bias)
    (mx mw mb : Admission.Metadata) (bx : Layout x mx) (bw : Layout w mw) (bb : Layout bias mb)
    (S D : Nat) (h : Admission.Contract mw mb mx S D) (scratch : State ℝ)
    (Y Mean Rstd : RegionName) (c : Configuration) (treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    let s := state read decode x w bias scratch
    let xv := logical read decode x lx
    let wv := logical read decode w lw
    let bv := logical read decode bias lb
    ∃ final,
      Source.launch (withEquality (ideal rest) (fun _ _ => Bool.false)) s Y Mean Rstd S D c.block
        (SourceBinding.tree c.tree) = some final ∧
      Observations read decode x w bias lx lw lb s S D ∧
      Native.SourceExact S D c.block ∧
      (∀ row, row < S → final.mem Mean row = .mk .real (Math.mean xv D row)) ∧
      (∀ row, row < S → final.mem Rstd row = .mk .real (Math.inverse xv D row)) ∧
      (∀ row, row < S → ∀ col, col < D → final.mem Y (col + D * row) =
        .mk .bf16 (Math.output xv wv bv D row col)) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) := by
  dsimp only
  let s := state read decode x w bias scratch
  have obs := observed read decode x w bias lx lw lb mx mw mb bx bw bb S D h scratch
  obtain ⟨final, run, hm, hi, hy, frame⟩ := KernelRealResult.kernel_real rest s
    (logical read decode x lx) (logical read decode w lw) (logical read decode bias lb)
    S D Y Mean Rstd c treeComplete sep obs.xValues obs.wValues obs.biasValues
  exact ⟨final, run, obs, Native.source_exact S D c.block ⟨h.width, h.extent⟩ c.allowed, hm, hi, hy, frame⟩

end Real

section Native
open NativeInputs

/-- Complete source execution and independent native outputs over original
logical Tensor words. The native profile sees only operation shape, constant
metadata and physical leaf identity, never raw input values. All compiler
choices, primitive arithmetic and conversions are explicit modeling boundaries. -/
theorem layer_norm_welfold_native (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (read : Nat → Word) (x w bias : VeriTile.Torch.FP32.Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : TensorLaws bias)
    (mx mw mb : Admission.Metadata) (bx : InputBinding.Layout x mx)
    (bw : InputBinding.Layout w mw) (bb : InputBinding.Layout bias mb)
    (S D : Nat) (h : Admission.Contract mw mb mx S D) (scratch : State NativeKernel.Term)
    (Y Mean Rstd : RegionName) (c : Configuration) (_treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    let s := state read x w bias scratch
    let result := NativeKernel.reference I P (logicalGraph read x lx)
      (logicalGraph read w lw) (logicalGraph read bias lb) D c.block
    ∃ final,
      Source.launch (withEquality NativeKernel.algebra (fun _ _ => Bool.false)) s Y Mean Rstd
        S D c.block (SourceBinding.tree c.tree) = some final ∧
      Observations read x w bias lx lw lb s S D ∧
      Native.SourceExact S D c.block ∧
      (∀ row, row < S → NativeKernel.observeCell I P (final.mem Mean row) = .mk .real (result row c.tree).mean) ∧
      (∀ row, row < S → NativeKernel.observeCell I P (final.mem Rstd row) = .mk .real (result row c.tree).inverse) ∧
      (∀ row, row < S → ∀ col, col < D → NativeKernel.observeCell I P (final.mem Y (col + D * row)) =
        .mk .bf16 ((result row c.tree).output col)) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off →
        NativeKernel.observeCell I P (final.mem q off) = NativeKernel.observeCell I P (s.mem q off)) := by
  dsimp only
  let s := state read x w bias scratch
  have obs := observed read x w bias lx lw lb mx mw mb bx bw bb S D h scratch
  obtain ⟨final, run, hm, hi, hy, frame, observedFrame⟩ := KernelNativeResult.kernel_native I P s
    (logicalGraph read x lx) (logicalGraph read w lw) (logicalGraph read bias lb)
    S D Y Mean Rstd c sep obs.xValues obs.wValues obs.biasValues
  exact ⟨final, run, obs, Native.source_exact S D c.block ⟨h.width, h.extent⟩ c.allowed,
    hm, hi, hy, frame, observedFrame⟩

/-- Actual FP32 byte inputs and actual output pointer arguments compose with
successful typed source execution. Every FP32 mean/rstd byte and every BF16 Y
byte is computed from the independent native graph; all other bytes, including
all caller input backing storage, are preserved. Output placement/freshness are
ordinary kernel buffer facts, not a modeled host allocation or return protocol. -/
theorem layer_norm_welfold_native_physical (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (before : Physical.Bytes) (x w bias : VeriTile.Torch.FP32.Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : TensorLaws bias)
    (mx mw mb : Admission.Metadata) (bx : InputBinding.Layout x mx)
    (bw : InputBinding.Layout w mw) (bb : InputBinding.Layout bias mb)
    (S D : Nat) (h : Admission.Contract mw mb mx S D) (scratch : State NativeKernel.Term)
    (Y Mean Rstd : RegionName) (c : Configuration) (_treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (pointers : Physical.Outputs S D) (fresh : Physical.FreshInputs pointers w bias x) :
    let read := Physical.read32 before
    let s := state read x w bias scratch
    let result := NativeKernel.reference I P (logicalGraph read x lx)
      (logicalGraph read w lw) (logicalGraph read bias lb) D c.block
    ∃ final,
      Source.launch (withEquality NativeKernel.algebra (fun _ _ => Bool.false)) s Y Mean Rstd
        S D c.block (SourceBinding.tree c.tree) = some final ∧
      Observations read x w bias lx lw lb s S D ∧
      Native.SourceExact S D c.block ∧
      (∀ row, row < S → ∀ byte, byte < 4 →
        Physical.outputBytes I P pointers Y Mean Rstd before final (pointers.mean.toNat + 4 * row + byte) =
          Physical.byteOf (result row c.tree).mean byte) ∧
      (∀ row, row < S → ∀ byte, byte < 4 →
        Physical.outputBytes I P pointers Y Mean Rstd before final (pointers.rstd.toNat + 4 * row + byte) =
          Physical.byteOf (result row c.tree).inverse byte) ∧
      (∀ row, row < S → ∀ col, col < D → ∀ byte, byte < 2 →
        Physical.outputBytes I P pointers Y Mean Rstd before final
          (pointers.y.toNat + 2 * (col + D * row) + byte) =
          Physical.byteOf ((result row c.tree).output col) byte) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) ∧
      (∀ address, Physical.Outside pointers address →
        Physical.outputBytes I P pointers Y Mean Rstd before final address = before address) ∧
      (∀ t, t = w ∨ t = bias ∨ t = x → ∀ i, i < t.storage.elements → ∀ byte, byte < 4 →
        Physical.outputBytes I P pointers Y Mean Rstd before final (t.storage.address i + byte) =
          before (t.storage.address i + byte)) := by
  dsimp only
  let read := Physical.read32 before
  let s := state read x w bias scratch
  have obs := observed read x w bias lx lw lb mx mw mb bx bw bb S D h scratch
  obtain ⟨final, run, hm, hi, hy, frame⟩ := KernelNativeResult.kernel_graph s
    (logicalGraph read x lx) (logicalGraph read w lw) (logicalGraph read bias lb) S D Y Mean Rstd c sep
    obs.xValues obs.wValues obs.biasValues
  refine ⟨final, run, obs, Native.source_exact S D c.block ⟨h.width, h.extent⟩ c.allowed,
    ?_, ?_, ?_, frame, Physical.frame I P pointers Y Mean Rstd before final, ?_⟩
  · intro row hr byte hb
    rw [Physical.mean_bytes I P pointers Y Mean Rstd before final row byte hr hb, hm row hr]
    rfl
  · intro row hr byte hb
    rw [Physical.rstd_bytes I P pointers Y Mean Rstd before final row byte hr hb, hi row hr]
    rfl
  · intro row hr col hc byte hb
    have bound : col + D * row < S * D := by nlinarith
    rw [Physical.output_bytes I P pointers Y Mean Rstd before final _ byte bound hb, hy row hr col hc]
    rfl
  · intro t ht i hi byte hb
    exact Physical.frame I P pointers Y Mean Rstd before final _ (fresh t ht i hi byte hb)

end Native

#axiomsClean python_checks_iff_preconditions
#axiomsClean layer_norm_welfold_real
#axiomsClean layer_norm_welfold_native
#axiomsClean layer_norm_welfold_native_physical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.TensorBridge
