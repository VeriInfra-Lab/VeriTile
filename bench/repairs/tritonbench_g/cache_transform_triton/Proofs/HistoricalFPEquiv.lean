import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Host
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Admission
import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Entry
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Native
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Schedule
import VeriTile.Triton.Float.StructuralIO

namespace VeriTile.Bench.Repairs.CacheTransform.HistoricalFPEquiv
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 100000

/-! Exact native-word helper for the selected FP32 cache pair.
Default source/test storage is FP32. No floating operation is executed:
metadata is signed int64, address arithmetic is uint64, payloads are uint8.
Every FP32 bit pattern is allowed; there are no finite-value conditions,
rounding identities, numerical ports or whole-program assumptions. -/

def nativeObserve (d : Storage) (word : Fin d.bytes → Nat) : List Nat := List.ofFn word

def numericalObligations : List String := []

theorem native_correct (M : Algebra Unit)
    (cos sin : Parameters) (s : State Unit) (mode : Mode) (lengths : List Nat)
    (cv sv : LogicalView (List Nat)) (width : Nat)
    (cs : DirectView cv width (nativeObserve cv.dtype)) (ss : DirectView sv width (nativeObserve sv.dtype))
    (cb : EntryBinding cos s mode lengths cv width (nativeObserve cv.dtype) cs)
    (sb : EntryBinding sin s mode lengths sv width (nativeObserve sv.dtype) ss)
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
    observedTensor t cos.output (preparedRows mode lengths).length width cv.dtype (nativeObserve cv.dtype) =
      mathematical mode lengths (fun r => fun c : Fin width => cv.logical r c.val) ∧
    observedTensor t sin.output (preparedRows mode lengths).length width sv.dtype (nativeObserve sv.dtype) =
      mathematical mode lengths (fun r => fun c : Fin width => sv.logical r c.val) ∧
    (∀ r o, (r ≠ cos.output ∨ o ≥ cos.count) → (r ≠ sin.output ∨ o ≥ sin.count) →
      t.mem r o = s.mem r o) :=
  whole_call M cos sin s mode lengths cv sv width (nativeObserve cv.dtype) (nativeObserve sv.dtype)
    cs ss cb sb validCos validSin freshOutput freshCos freshRows

#axiomsClean native_correct

/-- FP proof-calculus view of every output byte of the complete actual gather.
The discrete native channel uses exact equality, and the reference is the
independent indexed byte mapping. Whole-pair success, words and frame are
retained by native_correct above. -/
theorem byte_equivalence (M : Algebra (FP.Equational.Term Unit))
    (p : Parameters) (s : State (FP.Equational.Term Unit))
    (n capacity : Nat) (rows cache : Nat → Nat)
    (input : 0 < p.count → GatherInput p s n capacity rows cache)
    (o : Nat) (ho : o < p.count) :
    ValueRelated ([] : VeriTile.Spec.Assumptions ComputeStmt) .nat
      (((gather M p s).mem p.output o).read .nat) (byteTarget p rows cache o) := by
  change _ = _
  rw [(gather_correct_total M p s n capacity rows cache input).1 o ho, Cell.read_same]

#axiomsClean byte_equivalence
#print_fp_assumptions byte_equivalence

/-- Explicit checked FP32 CUDA contract, direct input Tensor views, signed
length admission, and precisely one launch per nonempty output. -/
theorem native_correct_guarded (M : Algebra Unit)
    (cos sin : Parameters) (s : State Unit) (request : Request) (admitted : checkInputs request=Bool.true)
    (cv sv : LogicalView (List Nat)) (width : Nat)
    (cs : DirectView cv width (nativeObserve cv.dtype)) (ss : DirectView sv width (nativeObserve sv.dtype))
    (cb : EntryBinding cos s request.mode request.naturalLengths cv width (nativeObserve cv.dtype) cs)
    (sb : EntryBinding sin s request.mode request.naturalLengths sv width (nativeObserve sv.dtype) ss)
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
    observedTensor t cos.output (preparedRows request.mode request.naturalLengths).length width cv.dtype (nativeObserve cv.dtype) =
      mathematical request.mode request.naturalLengths (fun r => fun c : Fin width => cv.logical r c.val) ∧
    observedTensor t sin.output (preparedRows request.mode request.naturalLengths).length width sv.dtype (nativeObserve sv.dtype) =
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
  · exact native_correct M cos sin s request.mode request.naturalLengths cv sv width cs ss cb sb
      (by simpa [binding.cosRows] using valid.1) (by simpa [binding.sinRows] using valid.2)
      freshOutput freshCos freshRows

#axiomsClean native_correct_guarded
#specNonCircular mathematical avoiding [source,gather,pairGather,runTiles]
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.CacheTransform.HistoricalFPEquiv
