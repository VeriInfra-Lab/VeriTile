import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Triton.Memory.KernelSpec.Grouped

/-! Grouped regional IO uses the shared typed UKernelIO judgment. All channels,
windows and masks are those of GroupedMasked2DKernelIO. The source executes with
its actual regional reads; no disjoint flat-allocation premise is introduced. -/
namespace VeriTile.Triton.GroupedMasked2DKernelIO

/-- The existing grouped public fields, embedded into the unified typed core. -/
def regionalCore (io : GroupedMasked2DKernelIO) (hout : ∀ o, io.out o ∈ io.bufs) :
    UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := io.nIn
  nOut := io.nOut
  nScr := 0
  bufs := io.bufs
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := io.out
  obuf_mem := hout
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ j => io.read i p₀ p₁ j
  imask := fun i _ p₀ p₁ _ j => io.readMask i p₀ p₁ j
  owin := fun o _ p₀ p₁ _ j => io.write o p₀ p₁ j
  omask := fun o _ p₀ p₁ _ j => io.writeMask o p₀ p₁ j
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Source access safety, successful regional execution, independently specified
outputs and the exact active-output frame. Output channels must occur in the
public buffer list; no allocation disjointness or zero-undef premise is needed. -/
def RegionalImplements (io : GroupedMasked2DKernelIO)
    (f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut → Fin io.B → ℝ) : Prop :=
  ∃ hout : ∀ o, io.out o ∈ io.bufs, (io.regionalCore hout).RegionalImplements f

scoped infix:25 " ⊨ᵣ " => GroupedMasked2DKernelIO.RegionalImplements

/-- Compose the existing source safety and regional value/frame results. -/
theorem RegionalImplements.intro (io : GroupedMasked2DKernelIO)
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut → Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i (s.pids 0) (s.pids 1) j →
        io.read i (s.pids 0) (s.pids 1) j < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o (s.pids 0) (s.pids 1) j →
        io.write o (s.pids 0) (s.pids 1) j < bounds (io.out o)) →
      Kernel.TraceSafe bounds io.kernel.toAlgKernel s)
    (hrun : ∀ (s : BlockState) (xs : Fin io.nIn → Fin io.B → ℝ),
      (∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i (s.pids 0) (s.pids 1) j →
        s.readMem (io.inp i) (io.read i (s.pids 0) (s.pids 1) j) = xs i j) →
      ∃ final, exec io.kernel.toAlgKernel s = some final ∧
        (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o (s.pids 0) (s.pids 1) j →
          final.readMem (io.out o) (io.write o (s.pids 0) (s.pids 1) j) =
            f (s.pids 0) (s.pids 1) xs o j) ∧
        (∀ r address, (∀ (o : Fin io.nOut) (j : Fin io.B),
          io.writeMask o (s.pids 0) (s.pids 1) j →
          r ≠ io.out o ∨ address ≠ io.write o (s.pids 0) (s.pids 1) j) →
          final.mem r address = s.mem r address)) : io.RegionalImplements f := by
  refine ⟨hout, ?_⟩
  unfold UKernelIO.RegionalImplements
  apply UKernelIO.RegionalImplements3D.intro (io.regionalCore hout)
  · intro bounds s vals _hin hib hob _hsb
    exact hts bounds s hib hob
  · intro s vals hin
    obtain ⟨final, he, hv, hf⟩ := hrun s vals hin
    exact ⟨final, he, hv, fun r address outside _ => hf r address outside⟩

/-- Recover the complete regional execution component, without capacity or
physical-separation assumptions. This supports actual input-binding adapters. -/
theorem RegionalImplements.run {io : GroupedMasked2DKernelIO}
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut → Fin io.B → ℝ}
    (h : io.RegionalImplements f) (s : BlockState) (xs : Fin io.nIn → Fin io.B → ℝ)
    (hin : ∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i (s.pids 0) (s.pids 1) j →
      s.readMem (io.inp i) (io.read i (s.pids 0) (s.pids 1) j) = xs i j) :
    ∃ final, exec io.kernel.toAlgKernel s = some final ∧
      (∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o (s.pids 0) (s.pids 1) j →
        final.readMem (io.out o) (io.write o (s.pids 0) (s.pids 1) j) =
          f (s.pids 0) (s.pids 1) xs o j) ∧
      (∀ r address, (∀ (o : Fin io.nOut) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) j →
        r ≠ io.out o ∨ address ≠ io.write o (s.pids 0) (s.pids 1) j) →
        final.mem r address = s.mem r address) := by
  obtain ⟨hout, h⟩ := h
  obtain ⟨final, he, hv, hf⟩ := (h s xs hin).2
  exact ⟨final, he, hv, fun r address outside =>
    hf r address outside (fun t => t.elim0)⟩

/-- Capacity bounds certify the actual source trace, independently of a
particular logical input witness. The witness is read from the initial memory. -/
theorem RegionalImplements.traceSafe {io : GroupedMasked2DKernelIO}
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut → Fin io.B → ℝ}
    (h : io.RegionalImplements f) (bounds : RegionBounds) (s : BlockState)
    (hi : ∀ (i : Fin io.nIn) (j : Fin io.B), io.readMask i (s.pids 0) (s.pids 1) j →
      io.read i (s.pids 0) (s.pids 1) j < bounds (io.inp i))
    (ho : ∀ (o : Fin io.nOut) (j : Fin io.B), io.writeMask o (s.pids 0) (s.pids 1) j →
      io.write o (s.pids 0) (s.pids 1) j < bounds (io.out o)) :
    Kernel.TraceSafe bounds io.kernel.toAlgKernel s := by
  obtain ⟨hout, h⟩ := h
  let vals : Fin io.nIn → Fin io.B → ℝ := fun i j =>
    s.readMem (io.inp i) (io.read i (s.pids 0) (s.pids 1) j)
  exact (h s vals (fun _ _ _ => rfl)).1 bounds ⟨hi, ho, fun t => t.elim0⟩

/-- The same regional certificate recovers the existing disjoint flat-memory
endpoint when the canonical source is flattenable. No source result is reproved. -/
theorem RegionalImplements.toFlat {io : GroupedMasked2DKernelIO}
    {f : Nat → Nat → (Fin io.nIn → Fin io.B → ℝ) → Fin io.nOut → Fin io.B → ℝ}
    (h : io.RegionalImplements f) (hok : io.kernel.toAlgKernel.FlattenOk) :
    io.Implements f := by
  apply Implements.intro io h.choose hok
  · exact h.traceSafe
  · exact h.run

end VeriTile.Triton.GroupedMasked2DKernelIO
