import VeriTile.Triton.Memory.KernelSpec.Masked

/-! Regional kernel IO, using the same typed channels and windows as the flat
IO family. The relation executes the actual projected source, proves its active
access safety, and preserves raw cells outside output/private scratch windows.
It imposes no separation between regions: overlapping read-only physical views
may be represented by their coherent regional observations. This is a regional
judgment, not an automatic transport theorem for aliased flat pointers. -/
namespace VeriTile.Triton
namespace UKernelIO

/-- Initial typed values at every active declared input address. -/
def RegionalInputs (io : UKernelIO) (s : BlockState) (vals : io.Ctx) : Prop :=
  ∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
    io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    (io.ity i).read s (io.ibuf i)
      (io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals i j

/-- Every active declared input/output/private-scratch window is in bounds. -/
def RegionalBounds (io : UKernelIO) (bounds : RegionBounds)
    (s : BlockState) (vals : io.Ctx) : Prop :=
  (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
    io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.ibuf i)) ∧
  (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
    io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.obuf o)) ∧
  (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
    io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
    io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.sbuf t))

/-- The exact raw-cell frame excludes only active output and scratch cells. -/
def RegionalFrame (io : UKernelIO) (s final : BlockState) (vals : io.Ctx) : Prop :=
  ∀ r address,
    (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
      io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      r ≠ io.obuf o ∨ address ≠ io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) →
    (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
      io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
      r ≠ io.sbuf t ∨ address ≠ io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j) →
    final.mem r address = s.mem r address

/-- Source access safety and successful regional execution with independently
specified typed outputs. Safety follows from the declared window bounds and is
proved about the actual evaluated source, not assumed as a window property.
Execution and outputs are total over bound logical inputs, even when no finite
capacity assignment is supplied; this preserves existing regional theorems. -/
def RegionalImplements3D (io : UKernelIO)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ (s : BlockState) (vals : io.Ctx), io.RegionalInputs s vals →
    (∀ bounds, io.RegionalBounds bounds s vals →
      Kernel.TraceSafe bounds io.kernel.toAlgKernel s) ∧
    ∃ final, exec io.kernel.toAlgKernel s = some final ∧
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        (io.oty o).read final (io.obuf o)
          (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
          f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
      io.RegionalFrame s final vals

/-- Two-coordinate formula convenience; all three launch coordinates still
index the source, input/output windows and masks. -/
def RegionalImplements (io : UKernelIO)
    (f : Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.RegionalImplements3D (fun p₀ p₁ _ => f p₀ p₁)

scoped infix:25 " ⊨ᵤᵣ " => UKernelIO.RegionalImplements

/-- Composition from source trace safety and the regional execution proof.
There is no task-supplied runner or unconstrained safety callback in the IO. -/
theorem RegionalImplements3D.intro (io : UKernelIO)
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier}
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (vals : io.Ctx),
      io.RegionalInputs s vals →
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.ibuf i)) →
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.obuf o)) →
      (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
        io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j < bounds (io.sbuf t)) →
      Kernel.TraceSafe bounds io.kernel.toAlgKernel s)
    (hrun : ∀ (s : BlockState) (vals : io.Ctx), io.RegionalInputs s vals →
      ∃ final, exec io.kernel.toAlgKernel s = some final ∧
        (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
          (io.oty o).read final (io.obuf o)
            (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
            f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
        io.RegionalFrame s final vals) : io.RegionalImplements3D f := by
  intro s vals hpins
  exact ⟨fun bounds hb => hts bounds s vals hpins hb.1 hb.2.1 hb.2.2,
    hrun s vals hpins⟩

/-- A regional certificate plus the existing source flattenability proof
recovers the stronger disjoint flat-allocation contract without reproving the
program. The flat corollary does not enlarge the regional alias guarantee. -/
theorem RegionalImplements3D.toFlat (io : UKernelIO)
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier}
    (h : io.RegionalImplements3D f) (hok : io.kernel.toAlgKernel.FlattenOk) :
    io.Implements3D f := by
  rw [implements3D_iff_at]
  intro s vals
  apply ImplementsAt.intro io s vals hok
  · intro bounds _ hpins hi ho hs
    exact (h s vals hpins).1 bounds ⟨hi, ho, hs⟩
  · intro _ hpins
    exact (h s vals hpins).2

end UKernelIO

namespace MaskedKernelIO₂

/-- The regional interpretation reuses the typed unified IO core. -/
def regionalCore (io : MaskedKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.out] ++ io.scratch.map Prod.fst
  ity := fun _ => .float
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun t => (io.scratch.get t).1
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | _ => fun j => io.read2 p₀ + j.val
  imask := fun _ _ p₀ _ _ j => io.mask p₀ j
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ p₀ _ _ j => io.mask p₀ j
  swin := fun t _ p₀ _ _ j => (io.scratch.get t).2 p₀ + j.val
  smask := fun _ _ p₀ _ _ j => io.mask p₀ j

/-- Masked two-input regional IO; overlapping read-only physical observations
are allowed. Source safety, successful execution, active outputs and the exact
raw-cell frame are all carried by the shared relation. -/
def RegionalImplements (io : MaskedKernelIO₂)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  io.regionalCore.RegionalImplements
    (fun _ _ vals _ j => f (fun k => vals (⟨0, by decide⟩ : Fin 2) k)
      (fun k => vals (⟨1, by decide⟩ : Fin 2) k) j)

scoped infix:25 " ⊨ᵣ " => MaskedKernelIO₂.RegionalImplements

/-- Assemble the named masked interface from the existing source safety and
regional result. No flat-allocation separation premise is introduced. -/
theorem RegionalImplements.intro (io : MaskedKernelIO₂)
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
    (∀ j : Fin io.B, io.mask s.pid j →
      io.read1 s.pid + j.val < bounds io.in1) →
    (∀ j : Fin io.B, io.mask s.pid j →
      io.read2 s.pid + j.val < bounds io.in2) →
    (∀ j : Fin io.B, io.mask s.pid j →
      io.write s.pid + j.val < bounds io.out) →
    (∀ p ∈ io.scratch, ∀ j : Fin io.B, io.mask s.pid j →
      p.2 s.pid + j.val < bounds p.1) →
    Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
    (∀ j : Fin io.B, io.mask s₀.pid j →
      s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
    (∀ j : Fin io.B, io.mask s₀.pid j →
      s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
    ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin io.B, io.mask s₀.pid j →
          s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys j)
      ∧ (∀ r o,
          (r ≠ io.out ∨
            ∀ j : Fin io.B, io.mask s₀.pid j →
              o ≠ io.write s₀.pid + j.val) →
          (∀ p ∈ io.scratch, r = p.1 →
            ∀ j : Fin io.B, io.mask s₀.pid j →
              o ≠ p.2 s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o)) :
    io.RegionalImplements f := by
  unfold RegionalImplements UKernelIO.RegionalImplements
  apply UKernelIO.RegionalImplements3D.intro io.regionalCore
  · intro bounds s vals _hpins hib hob hsb
    refine hts bounds s (fun j hj => hib (⟨0, by decide⟩ : Fin 2) j hj)
      (fun j hj => hib (⟨1, by decide⟩ : Fin 2) j hj)
      (fun j hj => hob (⟨0, by decide⟩ : Fin 1) j hj) ?_
    intro q hq j hj
    obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
    have h : (io.scratch.get u).2 (s.pids 0) + j.val
        < bounds (io.scratch.get u).1 := hsb u j hj
    rw [hu] at h
    exact h
  · intro s₀ vals hpins
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 2) j)
        (fun j => vals (⟨1, by decide⟩ : Fin 2) j)
        (fun j hj => hpins (⟨0, by decide⟩ : Fin 2) j hj)
        (fun j hj => hpins (⟨1, by decide⟩ : Fin 2) j hj)
    refine ⟨s1, hexec, fun _o j hj => hval j hj, ?_⟩
    intro r o' hoc hsc'
    refine hframe r o' ?_ ?_
    · by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hoc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
    · intro q hq hrq j hj
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      have h : r ≠ (io.scratch.get u).1 ∨
          o' ≠ (io.scratch.get u).2 (s₀.pids 0) + j.val :=
        hsc' u j hj
      rw [hu] at h
      rcases h with hne | hno
      · exact absurd hrq hne
      · exact hno

end MaskedKernelIO₂
end VeriTile.Triton
