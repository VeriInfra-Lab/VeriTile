import bench.tritonbench_g.matmul_tma.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def scatter {α : Type} (p : Parameters) (half : Bool) (s : State α)
    (values : CIndex p → α) : State α :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => t.write p.c (cOffset p i)
      (.mk (outputDType half).toTileDType (ofFloat (outputDType half) (values i)))) s

theorem scatter_memory_congr {α : Type} (l : List (CIndex p))
    (half : Bool) (values : CIndex p → α) (s u : State α) (h : s.mem = u.mem) :
    (l.foldl (fun t i => t.write p.c (cOffset p i)
      (.mk (outputDType half).toTileDType (ofFloat (outputDType half) (values i)))) s).mem =
    (l.foldl (fun t i => t.write p.c (cOffset p i)
      (.mk (outputDType half).toTileDType (ofFloat (outputDType half) (values i)))) u).mem := by
  induction l generalizing s u with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [State.write, h]

/-- Full execution of all three block pointers, two loads, typed dot, output
conversion and tile store. The RHS is derived directly from initial memory. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) :
    ∃ t, FP.Structural.exec (backend M D) (source p half) s = some t ∧
      t.mem = (scatter p half s (outputValues M D p half s)).mem := by
  cases half <;>
    simp [source, typed, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, Region.cast, BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets,
      TileShape.indexToList, backend, outputValues, aValues, bValues,
      aOffset, bOffset, cOffset, outputDType, scatter, ofFloat, toFloat]
  · exact scatter_memory_congr (half := Bool.false) _ _ _ _ rfl
  · exact scatter_memory_congr (half := Bool.true) _ _ _ _ rfl

def Post {α : Type} (p : Parameters) (half : Bool) (s t : State α)
    (values : CIndex p → α) : Prop :=
  (∀ i, t.mem p.c (cOffset p i) =
    .mk (outputDType half).toTileDType (ofFloat (outputDType half) (values i))) ∧
  (∀ R off, R ≠ p.c ∨ (∀ i : CIndex p, off ≠ cOffset p i) →
    t.mem R off = s.mem R off)

theorem post_of_memory {α : Type} (p : Parameters) (half : Bool) (s t : State α)
    (v : CIndex p → α) (hi : Function.Injective (cOffset p))
    (hm : t.mem = (scatter p half s v).mem) : Post p half s t v := by
  constructor
  · intro i
    rw [hm]
    exact State.scatter_readback p.c (cOffset p) _ s i hi
  · intro R off h
    rw [hm]
    apply State.scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ => h i

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α)
    (hi : Function.Injective (cOffset p)) :
    ∃ t, FP.Structural.exec (backend M D) (source p half) s = some t ∧
      Post p half s t (outputValues M D p half s) := by
  obtain ⟨t, he, hm⟩ := execution_memory M D p half s
  exact ⟨t, he, post_of_memory p half s t _ hi hm⟩

#axiomsClean execution_memory
#axiomsClean post_of_memory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
