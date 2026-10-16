import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryContract

/-! The bit-value contract uses arbitrary caller-supplied input patterns.
Realization requires the actual source channel to contain those patterns;
it does not require finiteness or exclude NaNs. The output formula uses the
independent exponent classifier, and the frame retains complete tagged cells.
These are complete typed interpreter executions, not native CUDA claims. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedBits BitClassification
noncomputable section

def InputBits {α : Type} (p : Parameters) (bits : Nat → α) (s : State α) : Prop :=
  ∀ x, x < p.size → s.mem p.input (x * p.inputStride) = .mk .real (bits x)

def Written (p : Parameters) (pid numCtas : Nat) (r : RegionName) (o : Nat) : Prop :=
  r = p.output ∧ ∃ x, SourceCovers p pid numCtas x ∧ o = x * p.outputStride

def Post {α : Type} (classify : α → Bool) (p : Parameters) (bits : Nat → α)
    (initial final : State α) : Prop :=
  (∀ x, SourceCovers p (initial.pids 0) (initial.numPids 0) x →
    final.mem p.output (x * p.outputStride) = .mk .bool (classify (bits x))) ∧
  (∀ r o, ¬Written p (initial.pids 0) (initial.numPids 0) r o →
    final.mem r o = initial.mem r o)

theorem sourceCovers_lt (p : Parameters) (pid numCtas x : Nat)
    (hx : SourceCovers p pid numCtas x) : x < p.size := by
  have h := (sourceCovers_iff p pid numCtas x).mp hx
  cases hb : p.oneTile with
  | «true» =>
      simp only [hb, ↓reduceIte] at h
      obtain ⟨_, _, hlt⟩ := h
      exact hlt
  | «false» =>
      simp only [hb, Bool.false_eq_true, if_false] at h
      obtain ⟨_, _, _, _, hlt⟩ := h
      exact hlt

theorem post_of_memory {α : Type} [Inhabited α] (classify : α → Bool)
    (p : Parameters) (bits : Nat → α) (initial final : State α)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hinput : InputBits p bits initial)
    (hmem : final.mem =
      sourceMemory classify p (initial.pids 0) (initial.numPids 0) initial.mem) :
    Post classify p bits initial final := by
  obtain ⟨hvalue, hframe⟩ := sourceMemory_contract classify p
    (initial.pids 0) (initial.numPids 0) hne hstride initial
  constructor
  · intro x hx
    rw [hmem, hvalue x hx,
      hinput x (sourceCovers_lt p (initial.pids 0) (initial.numPids 0) x hx),
      Cell.read_same]
  · intro r o hmiss
    rw [hmem]
    apply hframe r o
    by_cases hr : r = p.output
    · right
      intro x hx heq
      exact hmiss ⟨hr, x, hx, heq⟩
    · exact Or.inl hr

theorem original32_correct (rest : Algebra Float32Bits) (p : Parameters)
    (bits : Nat → Float32Bits) (initial : State Float32Bits)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hinput : InputBits p bits initial) :
    ∃ final, FP.Structural.exec (engine (backend32 rest)) (original32 p) initial = some final ∧
      Post finite32 p bits initial final := by
  obtain ⟨final, he, hm⟩ := original32_run rest p initial
  exact ⟨final, he, post_of_memory finite32 p bits initial final hne hstride hinput hm⟩

theorem original64_correct (rest : Algebra Float64Bits) (p : Parameters)
    (bits : Nat → Float64Bits) (initial : State Float64Bits)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hinput : InputBits p bits initial) :
    ∃ final, FP.Structural.exec (engine (backend64 rest)) (original64 p) initial = some final ∧
      Post finite64 p bits initial final := by
  obtain ⟨final, he, hm⟩ := original64_run rest p initial
  exact ⟨final, he, post_of_memory finite64 p bits initial final hne hstride hinput hm⟩

#axiomsClean sourceCovers_lt
#axiomsClean post_of_memory
#axiomsClean original32_correct
#axiomsClean original64_correct
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
