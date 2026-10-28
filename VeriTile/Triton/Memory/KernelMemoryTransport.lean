import VeriTile.Triton.Memory.KernelSpec.Masked
import VeriTile.Triton.Memory.KernelSpec.Scatter
import VeriTile.Triton.Memory.KernelSpec.Tile
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton

/-- A completed run's memory observation retains failure and every cell.
Register representations may differ, but termination, values and frame transfer. -/
theorem exists_memory_congr {a b : Option BlockState}
    (h : a.map BlockState.mem = b.map BlockState.mem)
    (P : BlockState → Prop)
    (respects : ∀ s t, s.mem = t.mem → (P s ↔ P t)) :
    (∃ s, a = some s ∧ P s) ↔ (∃ s, b = some s ∧ P s) := by
  cases a with
  | none => cases b <;> simp_all
  | some s =>
    cases b with
    | none => simp_all
    | some t =>
      simp only [Option.map_some, Option.some.injEq] at h
      constructor
      · rintro ⟨u, hu, hp⟩
        obtain rfl := Option.some.inj hu
        exact ⟨t, rfl, (respects s t h).mp hp⟩
      · rintro ⟨u, hu, hp⟩
        obtain rfl := Option.some.inj hu
        exact ⟨s, rfl, (respects s t h).mpr hp⟩

/-- Transfer the unchanged complete IO contract along actual flat execution. -/
theorem MaskedKernelIO₁.Implements.congr_memory (io : MaskedKernelIO₁)
    (kernel : ComputeKernel) (projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel)
    (same : ∀ (A : FlatAlloc) (s : BlockState),
      (exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem =
      (exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem)
    (f : (Fin io.B → ℝ) → Fin io.B → ℝ) :
    io.Implements f ↔ ({ io with kernel := kernel, projection := projection } : MaskedKernelIO₁).Implements f := by
  unfold Implements
  repeat' (apply forall_congr'; intro)
  apply exists_memory_congr (same _ _)
  intro s t hm
  simp only [BlockState.readMem, hm]

#axiomsClean MaskedKernelIO₁.Implements.congr_memory

/-- Transfer the unchanged complete IO contract along actual flat execution. -/
theorem MaskedKernelIO₂.Implements.congr_memory (io : MaskedKernelIO₂)
    (kernel : ComputeKernel) (projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel)
    (same : ∀ (A : FlatAlloc) (s : BlockState),
      (exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem =
      (exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
    io.Implements f ↔ ({ io with kernel := kernel, projection := projection } : MaskedKernelIO₂).Implements f := by
  unfold Implements
  repeat' (apply forall_congr'; intro)
  apply exists_memory_congr (same _ _)
  intro s t hm
  simp only [BlockState.readMem, hm]

#axiomsClean MaskedKernelIO₂.Implements.congr_memory

/-- Transfer the unchanged complete IO contract along actual flat execution. -/
theorem MetaScatterMasked2DKernelIO₁.Implements.congr_memory (io : MetaScatterMasked2DKernelIO₁)
    (kernel : ComputeKernel) (projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel)
    (same : ∀ (A : FlatAlloc) (s : BlockState),
      (exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem =
      (exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem)
    (f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) → (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ) :
    io.Implements f ↔ ({ io with kernel := kernel, projection := projection } : MetaScatterMasked2DKernelIO₁).Implements f := by
  unfold Implements
  repeat' (apply forall_congr'; intro)
  apply exists_memory_congr (same _ _)
  intro s t hm
  simp only [BlockState.readMem, hm]

#axiomsClean MetaScatterMasked2DKernelIO₁.Implements.congr_memory

/-- Transfer the unchanged complete IO contract along actual flat execution. -/
theorem MaskedTileKernelIO₁.Implements.congr_memory (io : MaskedTileKernelIO₁)
    (kernel : ComputeKernel) (projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel)
    (same : ∀ (A : FlatAlloc) (s : BlockState),
      (exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem =
      (exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState s)).map BlockState.mem)
    (f : Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) :
    io.Implements f ↔ ({ io with kernel := kernel, projection := projection } : MaskedTileKernelIO₁).Implements f := by
  unfold Implements
  repeat' (apply forall_congr'; intro)
  apply exists_memory_congr (same _ _)
  intro s t hm
  simp only [BlockState.readMem, hm]

#axiomsClean MaskedTileKernelIO₁.Implements.congr_memory

end VeriTile.Triton
