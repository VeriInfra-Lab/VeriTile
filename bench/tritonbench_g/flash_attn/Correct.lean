import bench.tritonbench_g.flash_attn.Proofs.Contracts
import bench.tritonbench_g.flash_attn.Proofs.RealWitness
import bench.tritonbench_g.flash_attn.Proofs.LegacyReal

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.FlashAttn
open VeriTile.Triton

/-- **Genuine GENERAL `O`-store correctness** (both causal cases). Dimension-parameterized
`flash_attn_genuine_output_compute_correct`. -/
theorem flash_attn_genuine_output_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (if IS_CAUSAL then
            flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx
          else
            flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx)))) := by
  exact flash_attn_genuine_output_compute_correct_general_legacyProof Q K V L O s IS_CAUSAL sm_scale stride_q_head SEQLEN BLOCK_M DIM BLOCK_N sqbs skbs svbs sobs sosl sod BS HEAD hDIM hBN hBM hBMlen hdvd hSEQ hHi hOL hundef

/-- **Genuine GENERAL `L`-store correctness** (both causal cases). Dimension-parameterized
`flash_attn_genuine_l_compute_correct`. -/
theorem flash_attn_genuine_l_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) IS_CAUSAL (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2) := by
  exact flash_attn_genuine_l_compute_correct_general_legacyProof Q K V L O s IS_CAUSAL sm_scale stride_q_head SEQLEN BLOCK_M DIM BLOCK_N sqbs skbs svbs sobs sosl sod BS HEAD hDIM hBN hBM hBMlen hdvd hSEQ hHi hOL hundef

/-- **Python case 1 (causal) GENERAL genuine closed-form correctness.** -/
theorem flash_attn_python_case1_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hAlign : (s.pids 0 + 1) * BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.true)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) Bool.true (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2)) := by
  exact flash_attn_python_case1_genuine_compute_correct_general_legacyProof Q K V L O s sm_scale stride_q_head SEQLEN BLOCK_M DIM BLOCK_N sqbs skbs svbs sobs sosl sod BS HEAD hDIM hBN hBM hBMlen hdvd hSEQ hAlign hOL hundef

/-- **Python case 2 (non-causal) GENERAL genuine closed-form correctness.** -/
theorem flash_attn_python_case2_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) Bool.false (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2)) := by
  exact flash_attn_python_case2_genuine_compute_correct_general_legacyProof Q K V L O s sm_scale stride_q_head SEQLEN BLOCK_M DIM BLOCK_N sqbs skbs svbs sobs sosl sod BS HEAD hDIM hBN hBM hBMlen hdvd hSEQ hOL hundef

open scoped VeriTile.Triton.StreamMasked3DKernelIO₃ₓ₂ in
/-- **The non-causal `⊨[R]` io headline.** For every rounding model `R` with
`R.round .fp16 = id`, the FlashAttention forward surface at
`IS_CAUSAL = false` implements, on its `StreamMasked3DKernelIO₃ₓ₂` signature,
the **ideal-ℝ online-softmax attention fold** over the three streamed tiles:
output lane `j = (i, d)` of `O` holds the base-2 attention ratio
(`flashIOOutSpec` = the existing `flashAttnOValueSpec` closed form restated on
the streams), and row `i` of `L` holds the genuine log-sum-exp
`log₂ (Σⱼ 2^scoreⱼ)` (`flashIOLSpec`). `O` is quantized on the `.fp16` grid
(`out1DType := .fp16`) and `L` is the unrounded `.real` statistics row.

**Hypothesis provenance** (all inherited from the exact non-causal headline
`flash_attn_python_case2_genuine_compute_correct_general`):

* `hfp16 : R.round .fp16 = id` — the file's declared fp16 modeling boundary
  (three in-body `Op.castFloat` sites plus the `.fp16` terminal store); the
  exact stack already treats these casts as the identity.
* `hDIM`/`hBN`/`hSEQ` (`0 < DIM`, `0 < BLOCK_N`, `0 < SEQLEN`) and
  `hdvd : BLOCK_N ∣ SEQLEN` shape the KV walk — `T = SEQLEN / BLOCK_N` full
  blocks, exactly the exact stack's side conditions.
* `hOL : O ≠ L` keeps the `O` tile store from clobbering the `L` row.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself — and `hpid₀ = 0` is gone (the Phase-1
generalization), so this holds at every program id.

**Scope disclosed**: the **causal** (`IS_CAUSAL = true`) io face is
deliberately deferred (see the section header); it stays on its exact
headline `flash_attn_python_case1_genuine_compute_correct_general`. -/
theorem flash_attn_io_correctness (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N stride_q_head : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hSEQ : 0 < SEQLEN) (hdvd : BLOCK_N ∣ SEQLEN)
    (hOL : O ≠ L) :
    flashAttnIO Q K V L O sm_scale sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N
        stride_q_head ⊨[R]
      fun _ _ _ xs ys zs =>
        (fun j => flashIOOutSpec BLOCK_M DIM BLOCK_N SEQLEN (SEQLEN / BLOCK_N)
            (Nat.div_pos (Nat.le_of_dvd hSEQ hdvd) hBN) (Nat.div_mul_cancel hdvd) hBN
            sm_scale xs ys zs j,
         fun i => flashIOLSpec BLOCK_M DIM BLOCK_N SEQLEN (SEQLEN / BLOCK_N) hDIM
            (Nat.div_pos (Nat.le_of_dvd hSEQ hdvd) hBN) (Nat.div_mul_cancel hdvd) hBN
            sm_scale xs ys zs i) := by
  exact flash_attn_io_correctness_legacyProof R hfp16 Q K V L O sm_scale sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N stride_q_head hDIM hBN hSEQ hdvd hOL

end VeriTile.Bench.TritonBenchG.FlashAttn

/-! Full-source real contracts for both typed programs. The independent
formula uses the source's exact decimal scale, source-used strides, causal
mask, and rounded-up physical KV window. The two-output reference scatters L
then O in source lane order, so arbitrary output aliasing remains covered.
Legacy theorem signatures and their historical hypotheses are imported above. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Correctness
open VeriTile.Triton TypedSource Formula RealExecution
noncomputable section

specification original_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionIO (kernel input output Bool.false Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel p s Q K V L O) := by
  rw [full_source_erasure]
  exact source_real_correct Bool.false Q K V L O p s hBN hhi

specification candidate_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionIO (kernel input output Bool.true Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel p s Q K V L O) := by
  rw [full_source_erasure]
  exact source_real_correct Bool.true Q K V L O p s hBN hhi

specification flat_correctness (swapped : Bool) (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionFlatIO (kernel input output swapped Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel A p s Q K V L O) := by
  rw [full_source_erasure]
  exact source_flat_correct A hd hcov swapped Q K V L O p s hb hundef hBN hhi

/-- Both complete typed sources have actual access safety, successful finite-flat
execution, both independent attention outputs and the full raw-cell frame.
The formula includes every physically loaded padded KV lane. Output addresses
may alias: the reference retains the source's L-then-O store order. Floating
casts are erased here; this is not an IEEE or native frontend theorem. -/
specification full_flat_correctness (swapped : Bool) (input : Storage)
    (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.FlatIO
      (actualKernel input swapped Q K V L O p.sm_scale
        stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
        stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
        stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
        stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
        BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel
      A p s Q K V L O) := by
  rw [actual_full_source_erasure]
  exact Contracts.full_flat A hd hcov swapped Q K V L O p s hb hundef hBN hhi

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Correctness
