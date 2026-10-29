/- Optional historical whole-grid/native transport results. These are not
required public kernel acceptance and primitive conformance is not a deferred
numerical experiment. -/
import bench.repairs.tritonbench_g.apply_penalty.Proofs.FP
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Binding

namespace VeriTile.Bench.Repairs.ApplyPenalty.LegacyNative
open FPProof
noncomputable section

/-- Fixed FP32 words and original fusion-enabled scalar schedule. No numerical
identity or finite-input condition is required for complete output and frame. -/
theorem apply_penalty_fp32 (ops : NativeOps) (a : Arguments) (h : InputContract a)
    (input : Inputs Word) :
    wrapperRun ops.scalar a input=.ok (gridRun ops.scalar a input (List.range a.batches)) ∧
    (∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      gridRun ops.scalar a input (List.range a.batches) (b*a.vocab+a.token k)=
        nativePenalty ops (input.logits (b*a.vocab+a.token k)) (input.presence b)
          (input.frequency b) (input.repetition b) (a.counts k)) ∧
    (∀ q,(∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      q≠b*a.vocab+a.token k) →
      gridRun ops.scalar a input (List.range a.batches) q=input.logits q) := by
  have result := complete_grid ops.scalar a h input _ (List.Perm.refl _)
  exact ⟨by simp [wrapperRun,(checkInputs_iff a).mpr h],result.1,result.2⟩

/-- Optional exact-arithmetic corollary. Its strong exactness hypotheses are
not claimed throughout the admitted domain and are not experimental targets. -/
theorem apply_penalty_real_if_local (ops : NativeOps) (a : Arguments) (h : InputContract a)
    (input : Inputs Word) (decode : Word → ℝ) (finite : Word → Prop)
    (domain : NumericalDomain ops a input decode finite) :
    ∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      decode (gridRun ops.scalar a input (List.range a.batches) (b*a.vocab+a.token k))=
        penalty (decode (input.logits (b*a.vocab+a.token k))) (a.counts k).toInt
          (decode (input.presence b)) (decode (input.frequency b)) (decode (input.repetition b)) := by
  intro b hb k hk he
  obtain ⟨hl,eq⟩ := cover a h b k hb hk he
  have ev := (event_iff _ _ _).mp hl
  have site := domain b hb (k-a.start b) ev.1
  simp only [eq,he,if_true,address] at site
  rw [(apply_penalty_fp32 ops a h input).2.1 b hb k hk he]
  apply native_local_real ops decode finite
  simpa only [address,eq] using site

/-- Rounded native conformance uses only the local evaluated primitive sites;
the structural wrapper contract and unconditional frames are unchanged. -/
theorem apply_penalty_rounded_if_local (ops : NativeOps) (profile : RoundedProfile)
    (a : Arguments) (h : InputContract a) (input : Inputs Word)
    (domain : ∀ b,b<a.batches → ∀ lane,lane<a.block →
      LocalRounded ops profile
        (if a.start b+lane<a.stop b then input.logits (address a b lane) else ops.zero)
        (input.presence b) (input.frequency b) (input.repetition b)
        (if a.start b+lane<a.stop b then a.counts (a.start b+lane) else 0)) :
    ∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      gridRun ops.scalar a input (List.range a.batches) (b*a.vocab+a.token k)=
        nativePenalty (roundedOps profile) (input.logits (b*a.vocab+a.token k))
          (input.presence b) (input.frequency b) (input.repetition b) (a.counts k) := by
  intro b hb k hk he
  obtain ⟨hl,eq⟩ := cover a h b k hb hk he
  have site := domain b hb (k-a.start b) ((event_iff _ _ _).mp hl).1
  simp only [eq,he,if_true,address] at site
  rw [(apply_penalty_fp32 ops a h input).2.1 b hb k hk he]
  apply native_local_rounded ops profile
  simpa only [address,eq] using site

/-- Actual in-place byte stores on the supplied allocation. A codec contributes
only its local packing roundtrip. Entry logits are bound directly to that heap;
no copied-input/output snapshot or assumed kernel result is used. -/
theorem apply_penalty_bytes {Byte : Type} (codec : Physical.Codec Word Byte)
    (integer : Physical.Codec (BitVec 32) Byte)
    (ops : NativeOps) (a : Arguments) (h : InputContract a) (input : Inputs Word)
    (heap : Nat → Byte)
    (binding : Physical.ReadOnlyInputs codec integer a input heap)
    (logits : input.logits=Physical.read codec (a.tensor 0).dataPtr.toNat heap) :
    (∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      Physical.read codec (a.tensor 0).dataPtr.toNat
        (Physical.nativeRows codec integer ops.scalar a input (List.range a.batches) heap) (b*a.vocab+a.token k)=
      nativePenalty ops (input.logits (b*a.vocab+a.token k)) (input.presence b)
        (input.frequency b) (input.repetition b) (a.counts k)) ∧
    (∀ q,(∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b → ∀ byte : Fin 4,
      q≠(a.tensor 0).dataPtr.toNat+4*(b*a.vocab+a.token k)+byte.val) →
      Physical.nativeRows codec integer ops.scalar a input (List.range a.batches) heap q=heap q) ∧
    (∀ r : Fin 7,0<r.val → ∀ k,k<a.capacity r → ∀ byte : Fin 4,
      Physical.nativeRows codec integer ops.scalar a input (List.range a.batches) heap
        ((a.tensor r).dataPtr.toNat+4*k+byte.val)=heap ((a.tensor r).dataPtr.toNat+4*k+byte.val)) := by
  simp only [Physical.native_rows_eq codec integer ops.scalar a h input _
    (fun _ hb => List.mem_range.mp hb) heap binding]
  refine ⟨?_,?_,fun r hr k hk byte => Physical.metadata_frame codec ops.scalar a h input heap r hr k hk byte⟩
  · intro b hb k hk he
    rw [Physical.rows_read,← logits]
    exact (apply_penalty_fp32 ops a h input).2.1 b hb k hk he
  · intro q miss
    apply Physical.rows_frame
    intro b hb lane hl byte
    exact miss b (List.mem_range.mp hb) _ (by omega) ((event_iff _ _ _).mp hl).2 byte

#axiomsClean apply_penalty_fp32
#axiomsClean apply_penalty_real_if_local
#axiomsClean apply_penalty_rounded_if_local
#axiomsClean apply_penalty_bytes
end
end VeriTile.Bench.Repairs.ApplyPenalty.LegacyNative
