import bench.tritonbench_g.rotary_emb.Proofs.Witness
import bench.tritonbench_g.rotary_emb.Proofs.Execution

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


/-! Full real correctness of `_rotary_kernel` and the existing candidate.
The contract executes the complete source once and observes both buffers in
the same final state. For each active pair, the even result is
`x_even * cos_even - x_odd * sin_even`, and the odd result is
`x_even * sin_odd + x_odd * cos_odd`, using the initial input memory.

Q/K/Cos/Sin storage choices, shapes, strides and both program IDs are symbolic.
Sine uses the source's cosine strides. Both Q stores precede the K loads and
trigonometric reloads. Bounds include sequence-active trigonometric accesses
even with no active heads. Every cell outside the four actual stores is framed.

Necessary conditions are region separation, collision-free active output
addresses and active-access bounds. The flat-memory clause additionally states
the existing bridge's zero-undef condition. `Proofs/Witness.lean` proves a
nonempty, multisequence, multihead, strided flat execution. The real model erases
floating precision; FPEquiv.lean separately proves the typed candidate change.
These typed AST profiles do not imply native compiler acceptance: the unchanged
homogeneous fp8 source fails lowering in the recorded Triton 3.7.1 sm90 probe.
Native launch admissibility, machine-index overflow and composition across
programs remain backend/host boundaries. In particular the Python head grid
uses HEAD_Q and cannot cover K heads beyond the launched head blocks.
-/
namespace VeriTile.Bench.TritonBenchG.RotaryEmb.Correctness
open VeriTile Triton FullMath FullIO
open scoped FullIO

/-- Interleaved pair rotation from the initial data and trigonometric memory.
Odd and even results deliberately read their own trigonometric coordinates.
-/
noncomputable def rotaryPairSpec (p : Parameters) (X Cos Sin : RegionName)
    (offset : BlockState → Bool → p.Lane → Nat) : Formula p := fun s odd i =>
  let x0 := s.readMem X (offset s Bool.false i)
  let x1 := s.readMem X (offset s Bool.true i)
  let c := s.readMem Cos (p.trigAddress s odd i)
  let sn := s.readMem Sin (p.trigAddress s odd i)
  if odd then x0 * sn + x1 * c else x0 * c - x1 * sn

noncomputable def rotarySpec (p : Parameters) (Q K Cos Sin : RegionName) :
    Formula p × Formula p :=
  (rotaryPairSpec p Q Cos Sin p.qAddress, rotaryPairSpec p K Cos Sin p.kAddress)

/-- Independent real contracts for original and candidate, with successful
execution, the complete pair formula, source-mask safety and the full frame.
-/
specification rotary_emb_correctness (p : Parameters)
    (q k cos sin : FullSource.Storage) (Q K Cos Sin : RegionName) :
    Spec.Real (sourceIO p q k cos sin Bool.false Q K Cos Sin ⊨
      rotarySpec p Q K Cos Sin) ∧
    Spec.Real (sourceIO p q k cos sin Bool.true Q K Cos Sin ⊨
      rotarySpec p Q K Cos Sin) :=
  ⟨source_implements p q k cos sin Bool.false Q K Cos Sin,
    source_implements p q k cos sin Bool.true Q K Cos Sin⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.Correctness

/- The following historical headline concerns four isolated store slices.
Its original documentation, statement and proof are retained verbatim for
coverage discovery; the complete-source contract is stated above. -/
namespace VeriTile.Bench.TritonBenchG.RotaryEmb
open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

/-- **The headline for `rotary_emb_fwd`.** For arbitrary symbolic strides,
sequence length, head counts, and block sizes: the full `_rotary_kernel`
surface lowers to the algorithm layer, and each of the four Python-observable
in-place stores (Q even, Q odd, K even, K odd) implements the interleaved
rotary rotation on its grouped IO signature — for every disjoint flat placement
of `[Q/K, Cos, Sin]`, every program id whose active lanes are in bounds, and
every launch state whose four read windows hold `xs`, the translated pointer
kernel terminates, every write-active lane of the data buffer ends up holding
`dataEven·cos - dataOdd·sin` (even store) resp. `dataEven·sin + dataOdd·cos`
(odd store) computed from the **old** window contents, and every other memory
cell is unchanged. The four `dim ↦ dim · stride_d` injectivity hypotheses are
the within-family (not even-vs-odd) collision-freedom of the host layout. -/
theorem rotary_emb_kernel_correctness
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL BLOCK_HALF : Nat)
    (hQEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_qd))
    (hQOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd))
    (hKEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_kd))
    (hKOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd)) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    (rotaryQ0IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane) ∧
    (rotaryQ1IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane) ∧
    (rotaryK0IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * cosLane - dataOdd * sinLane) ∧
    (rotaryK1IO K Cos Sin stride_kbs stride_kh stride_kd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_K BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven := xs (⟨0, by decide⟩ : Fin 4) j
          let dataOdd := xs (⟨1, by decide⟩ : Fin 4) j
          let cosLane := xs (⟨2, by decide⟩ : Fin 4) j
          let sinLane := xs (⟨3, by decide⟩ : Fin 4) j
          dataEven * sinLane + dataOdd * cosLane) :=
  ⟨rotary_kernel_surface_toAlgorithm_supported Q K Cos Sin stride_qbs stride_qh
      stride_qd stride_kbs stride_kh stride_kd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL,
   rotary_emb_q0_block_implements Q Cos Sin stride_qbs stride_qh stride_qd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
      BLOCK_HALF hQEven,
   rotary_emb_q1_block_implements Q Cos Sin stride_qbs stride_qh stride_qd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q
      BLOCK_HALF hQOdd,
   rotary_emb_k0_block_implements K Cos Sin stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
      BLOCK_HALF hKEven,
   rotary_emb_k1_block_implements K Cos Sin stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len HEAD_K
      BLOCK_HALF hKOdd⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb
