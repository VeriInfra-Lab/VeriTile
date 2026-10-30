import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Host
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Admission
import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Ideal
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Entry
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Native
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Schedule

namespace VeriTile.Bench.Repairs.CacheTransform.HistoricalCorrect
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 100000

/-! Public Real contract. The source is the exact byte program in Kernels.
EntryBinding exposes the two byte windows, int64 metadata, fresh-output
separation, native extents and ordinary direct Tensor observation. `mathematical`
is independent sequence tensor math. All rows/columns and empty dimensions
are included. Finite IEEE words have their usual exact Real observation;
native special-value payloads are handled by FPEquiv, without importing it. -/

theorem real_correct (M : Algebra ℝ)
    (cos sin : Parameters) (s : State ℝ) (mode : Mode) (lengths : List Nat)
    (cv sv : LogicalView ℝ) (width : Nat)
    (cosObserve : (Fin cv.dtype.bytes → Nat) → ℝ)
    (sinObserve : (Fin sv.dtype.bytes → Nat) → ℝ)
    (cs : DirectView cv width cosObserve) (ss : DirectView sv width sinObserve)
    (cb : EntryBinding cos s mode lengths cv width cosObserve cs)
    (sb : EntryBinding sin s mode lengths sv width sinObserve ss)
    (validCos : ValidLengths mode lengths cv.rows)
    (validSin : ValidLengths mode lengths sv.rows)
    (freshOutput : cos.output ≠ sin.output)
    (freshCos : sin.cache ≠ cos.output) (freshRows : sin.rows ≠ cos.output) :
    let t := pairGather M cos sin s
    (∀ k, k < blocks cos.count →
      FP.Structural.exec M (source (tileParameters cos k))
        (tileState (runTiles M cos s k) k) = some (runTiles M cos s (k + 1))) ∧
    (∀ k, k < blocks sin.count →
      FP.Structural.exec M (source (tileParameters sin k))
        (tileState (runTiles M sin (gather M cos s) k) k) =
          some (runTiles M sin (gather M cos s) (k + 1))) ∧
    observedTensor t cos.output (preparedRows mode lengths).length width cv.dtype cosObserve =
      mathematical mode lengths (fun r => fun c : Fin width => cv.logical r c.val) ∧
    observedTensor t sin.output (preparedRows mode lengths).length width sv.dtype sinObserve =
      mathematical mode lengths (fun r => fun c : Fin width => sv.logical r c.val) ∧
    (∀ r o, (r ≠ cos.output ∨ o ≥ cos.count) → (r ≠ sin.output ∨ o ≥ sin.count) →
      t.mem r o = s.mem r o) :=
  whole_call M cos sin s mode lengths cv sv width cosObserve sinObserve cs ss cb sb
    validCos validSin freshOutput freshCos freshRows

#axiomsClean real_correct

/-- Explicit checked FP32 CUDA contract, direct input Tensor views, signed
length admission, and precisely one launch per nonempty output. -/
theorem real_correct_guarded (M : Algebra ℝ)
    (cos sin : Parameters) (s : State ℝ) (request : Request) (admitted : checkInputs request=Bool.true)
    (cv sv : LogicalView ℝ) (width : Nat)
    (cosObserve : (Fin cv.dtype.bytes → Nat) → ℝ)
    (sinObserve : (Fin sv.dtype.bytes → Nat) → ℝ)
    (cs : DirectView cv width cosObserve) (ss : DirectView sv width sinObserve)
    (cb : EntryBinding cos s request.mode request.naturalLengths cv width cosObserve cs)
    (sb : EntryBinding sin s request.mode request.naturalLengths sv width sinObserve ss)
    (binding : ViewBinding request cv sv width)
    (caller cosAllocation sinAllocation : Nat)
    (freshOutput : cos.output ≠ sin.output)
    (freshCos : sin.cache ≠ cos.output) (freshRows : sin.rows ≠ cos.output) :
    let result := returned M cos sin s request cv sv caller cosAllocation sinAllocation
    (result.cos=allocatedOutput cv request.mode request.naturalLengths cosAllocation ∧
      result.sin=allocatedOutput sv request.mode request.naturalLengths sinAllocation ∧
      result.restoredDevice=caller) ∧
    (∀ k,k<blocks cos.count → (tileParameters cos k).start=0 ∧ localPid k=k) ∧
    (∀ k,k<blocks sin.count → (tileParameters sin k).start=0 ∧ localPid k=k) ∧
    let t := pairGather M cos sin s
    (∀ k, k < blocks cos.count →
      FP.Structural.exec M (source (tileParameters cos k))
        (tileState (runTiles M cos s k) k) = some (runTiles M cos s (k + 1))) ∧
    (∀ k, k < blocks sin.count →
      FP.Structural.exec M (source (tileParameters sin k))
        (tileState (runTiles M sin (gather M cos s) k) k) =
          some (runTiles M sin (gather M cos s) (k + 1))) ∧
    observedTensor t cos.output (preparedRows request.mode request.naturalLengths).length width cv.dtype cosObserve =
      mathematical request.mode request.naturalLengths (fun r => fun c : Fin width => cv.logical r c.val) ∧
    observedTensor t sin.output (preparedRows request.mode request.naturalLengths).length width sv.dtype sinObserve =
      mathematical request.mode request.naturalLengths (fun r => fun c : Fin width => sv.logical r c.val) ∧
    (∀ r o, (r ≠ cos.output ∨ o ≥ cos.count) → (r ≠ sin.output ∨ o ≥ sin.count) →
      t.mem r o = s.mem r o) := by
  have h := (checkInputs_iff request).mp admitted
  have valid := admitted_lengths request h
  refine ⟨⟨rfl,rfl,rfl⟩,?_,?_,?_⟩
  · intro k hk
    exact entry_single_launch request h cos s cv width _ cs cb binding.width_eq k hk
  · intro k hk
    exact entry_single_launch request h sin s sv width _ ss sb binding.width_eq k hk
  · exact real_correct M cos sin s request.mode request.naturalLengths cv sv width cosObserve sinObserve cs ss cb sb
      (by simpa [binding.cosRows] using valid.1) (by simpa [binding.sinRows] using valid.2)
      freshOutput freshCos freshRows

#axiomsClean real_correct_guarded

/-- Complete mathematical call on arbitrary Real cache payloads, projected
from the byte source's proved per-word geometry. No Real-to-byte encoding,
finite representation, input-value assumption or output premise is used. -/
theorem whole_mathematical_real (request : Request) (checked : checkInputs request=Bool.true)
    (cos sin : RegionName) (fresh : cos≠sin) (cosInput sinInput : Nat → ℝ)
    (s : State ℝ) (caller : Nat) :
    let result := Ideal.call cos sin request.mode request.naturalLengths request.width cosInput sinInput s caller
    (∀ row∈preparedRows request.mode request.naturalLengths,row<request.cosRows ∧ row<request.sinRows) ∧
    result.value.1=mathematical request.mode request.naturalLengths
      (fun r => fun c : Fin request.width => cosInput (r*request.width+c.val)) ∧
    result.value.2=mathematical request.mode request.naturalLengths
      (fun r => fun c : Fin request.width => sinInput (r*request.width+c.val)) ∧
    (∀ region q,(region≠cos ∨ request.outputRows*request.width ≤ q) →
      (region≠sin ∨ request.outputRows*request.width ≤ q) → result.state.mem region q=s.mem region q) ∧
    result.objects=(cos,sin) ∧ result.device=caller := by
  have h := (checkInputs_iff request).mp checked
  have valid := admitted_lengths request h
  refine ⟨?_,?_⟩
  · intro row member
    exact ⟨preparedRows_bound _ _ _ valid.1 row member,preparedRows_bound _ _ _ valid.2 row member⟩
  · simpa only [outputRows_eq] using Ideal.pair_math cos sin fresh request.mode request.naturalLengths
      request.width cosInput sinInput s caller

#axiomsClean whole_mathematical_real
#specNonCircular mathematical avoiding [Ideal.call,Ideal.run,source,gather,pairGather]
#specNonCircular mathematical avoiding [source,gather,pairGather,runTiles]
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.CacheTransform.HistoricalCorrect
