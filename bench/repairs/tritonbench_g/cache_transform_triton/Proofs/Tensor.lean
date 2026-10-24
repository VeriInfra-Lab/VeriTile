import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 100000

def tableRows (mode : Mode) (lengths : List Nat) (r : Nat) : Nat :=
  ((preparedRows mode lengths)[r]?).getD 0

theorem tableRows_get (mode : Mode) (lengths : List Nat) (r : Nat)
    (hr : r < (preparedRows mode lengths).length) :
    tableRows mode lengths r = (preparedRows mode lengths)[r] := by
  simp [tableRows, hr]

theorem tableRows_bound (mode : Mode) (lengths : List Nat) (capacity : Nat)
    (h : ValidLengths mode lengths capacity) (r : Nat)
    (hr : r < (preparedRows mode lengths).length) :
    tableRows mode lengths r < capacity := by
  rw [tableRows_get mode lengths r hr]
  exact preparedRows_bound mode lengths capacity h _ (List.getElem_mem hr)

theorem storage_bytes_positive (d : Storage) : 0 < d.bytes := by cases d <;> decide

def observedTensor {α β : Type} [Inhabited α] (s : State α) (out : RegionName)
    (n width : Nat) (d : Storage) (observe : (Fin d.bytes → Nat) → β) : List (Fin width → β) :=
  List.ofFn (fun r : Fin n => fun c =>
    observe (fun b => (s.mem out ((r.val * width + c.val) * d.bytes + b.val)).read .nat))

def observedCache {β : Type} (cache : Nat → Nat) (width : Nat) (d : Storage)
    (observe : (Fin d.bytes → Nat) → β) (r : Nat) : Fin width → β :=
  fun c => observe (fun b => cache ((r * width + c.val) * d.bytes + b.val))

/-- This bridges actual complete output storage to independent tensor math,
including zero width: equality of empty row vectors is vacuous, while the
number of rows is retained. No successful-output premise crosses into a
numerical port; callers obtain `hout` from `pair_correct`. -/
theorem tensor_from_bytes {α β : Type} [Inhabited α] (p : Parameters) (t : State α)
    (mode : Mode) (lengths : List Nat) (cache : Nat → Nat) (width : Nat) (d : Storage)
    (observe : (Fin d.bytes → Nat) → β)
    (hshape : p.count = (preparedRows mode lengths).length * (width * d.bytes))
    (hwidth : p.rowBytes = width * d.bytes)
    (hout : ∀ o, o < p.count → t.mem p.output o =
      .mk .nat (byteTarget p (tableRows mode lengths) cache o)) :
    observedTensor t p.output (preparedRows mode lengths).length width d observe =
      mathematical mode lengths (observedCache cache width d observe) := by
  rw [← preparedRows_math]
  apply List.ext_getElem
  · simp [observedTensor]
  · intro i h₁ h₂
    have hi : i < (preparedRows mode lengths).length := by
      simpa only [List.length_map] using h₂
    simp only [observedTensor, List.getElem_ofFn, List.getElem_map]
    funext col
    have h := word_observation p t (tableRows mode lengths) cache
      (preparedRows mode lengths).length width d.bytes hshape hwidth
      (storage_bytes_positive d) hout observe i col.val hi col.isLt
    rw [tableRows_get mode lengths i hi] at h
    exact h

/-- The byte view reads the admitted contiguous input Tensor directly. Its
coordinate equation is an ordinary Tensor observation for every memory state;
it contains no copy, output, kernel execution or numerical result. -/
structure DirectView {β : Type} (v : LogicalView β) (width : Nat)
    (observe : (Fin v.dtype.bytes → Nat) → β) where
  bytes : Nat → Nat
  width_eq : width = v.width
  byteRange : ∀ o, o < v.rows * (width * v.dtype.bytes) → bytes o < 256
  coordinates : ∀ r, r < v.rows → ∀ c : Fin width,
    observedCache bytes width v.dtype observe r c = v.logical r c.val

theorem direct_view_math {β : Type} (v : LogicalView β) (width : Nat)
    (observe : (Fin v.dtype.bytes → Nat) → β) (direct : DirectView v width observe)
    (mode : Mode) (lengths : List Nat) (valid : ValidLengths mode lengths v.rows) :
    mathematical mode lengths (observedCache direct.bytes width v.dtype observe) =
      mathematical mode lengths (fun r => fun c : Fin width => v.logical r c.val) := by
  rw [← preparedRows_math, ← preparedRows_math]
  apply List.map_congr_left
  intro r hr
  funext c
  exact direct.coordinates r (preparedRows_bound mode lengths v.rows valid r hr) c

#axiomsClean tensor_from_bytes
#axiomsClean direct_view_math
end
end VeriTile.Bench.Repairs.CacheTransform
