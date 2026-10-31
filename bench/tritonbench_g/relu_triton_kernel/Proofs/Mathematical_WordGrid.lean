import bench.tritonbench_g.relu_triton_kernel.Kernels
import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_IntegerWords

/-! Input-word relation and composition with actual typed ReLU grid execution.
The relation binds every active input element to a finite source-width word.
It is an explicit hypothesis about logical cells, not an allocator or native
byte-load theorem. The output of every write is tagged fp32. The relation
theorem does not require distinct logical regions; physical mixed-width
aliases, byte offsets, alignment and concurrency are outside this model. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.IntegerWords
open VeriTile.Triton FP.Structural GatedGrid TypedValue
noncomputable section

def InputWords {α : Type} (e : Encoding) (words : Nat → Word e)
    (x : RegionName) (N : Nat) (mem : Memory α) : Prop :=
  ∀ o, o < N → mem x o = decodeCell e (words o)

/-- A concrete logical memory constructor. The baseline is arbitrary outside
the active input. A disjoint output region is preserved by a separate lemma. -/
def withInputWords {α : Type} (e : Encoding) (words : Nat → Word e)
    (x : RegionName) (N : Nat) (initial : State α) : State α :=
  { initial with mem := fun r o =>
      if r = x ∧ o < N then decodeCell e (words o) else initial.mem r o }

theorem withInputWords_encoded {α : Type} (e : Encoding) (words : Nat → Word e)
    (x : RegionName) (N : Nat) (initial : State α) :
    InputWords e words x N (withInputWords e words x N initial).mem := by
  intro o ho
  simp [withInputWords, ho]

theorem separated_output_preserved {α : Type} (e : Encoding) (words : Nat → Word e)
    (x y : RegionName) (N : Nat) (initial : State α) (separate : y ≠ x) (o : Nat) :
    (withInputWords e words x N initial).mem y o = initial.mem y o := by
  simp [withInputWords, separate]

/-- Initial-word result independent of executions and logical cell decoding.
Only the original pid-zero tile writes. The entire remaining memory is the
arbitrary initial memory, including an unfilled output tail. -/
def wordReference {α : Type} (M : Algebra α) (e : Encoding)
    (words : Nat → Word e) (y : RegionName) (N B : Nat)
    (initial : Memory α) : Memory α :=
  fun r o => if r = y ∧ o < B ∧ o < N then
    .mk .fp32 (wordValue M e (words o)) else initial r o

def wordSpec {α : Type} (M : Algebra α) (e : Encoding)
    (words : Nat → Word e) (y : RegionName) (N B : Nat)
    (initial : Memory α) : Memory α :=
  if gridCount N B = 0 then initial else wordReference M e words y N B initial

theorem reference_refines {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (e : Encoding) (words : Nat → Word e)
    (x y : RegionName) (N B : Nat) (initial : Memory α)
    (encoded : InputWords e words x N initial) :
    reference M cmp16 cmp32 e.profile.kind x y N B initial =
      wordReference M e words y N B initial := by
  funext r o
  by_cases active : r = y ∧ o < B ∧ o < N
  · simp only [reference, wordReference, active]
    rw [encoded o active.2.2, scalar_refines]
  · simp [reference, wordReference, active]

theorem wordSpec_frame {α : Type} (M : Algebra α) (e : Encoding)
    (words : Nat → Word e) (y : RegionName) (N B : Nat) (initial : Memory α)
    (r : RegionName) (o : Nat) (outside : r ≠ y ∨ B ≤ o ∨ N ≤ o) :
    wordSpec M e words y N B initial r o = initial r o := by
  unfold wordSpec
  split
  · rfl
  · unfold wordReference
    rw [if_neg]
    intro active
    rcases outside with hr | hb | hn
    · exact hr active.1
    · omega
    · omega

theorem wordSpec_written_fp32 {α : Type} (M : Algebra α) (e : Encoding)
    (words : Nat → Word e) (y : RegionName) (N B : Nat) (initial : Memory α)
    (o : Nat) (inBlock : o < B) (inInput : o < N) :
    wordSpec M e words y N B initial y o = .mk .fp32 (wordValue M e (words o)) := by
  have hn : 0 < N := by omega
  have hb : 0 < B := by omega
  have count := positive_grid_count N B hn hb
  simp [wordSpec, wordReference, Nat.ne_of_gt count, inBlock, inInput]

/-- If the prelaunch output region is tagged fp32, its untouched tail keeps
that tag. This hypothesis does not assert fresh or disjoint allocation. -/
theorem wordSpec_output_tag {α : Type} (M : Algebra α) (e : Encoding)
    (words : Nat → Word e) (y : RegionName) (N B : Nat) (initial : Memory α)
    (initialTag : ∀ o, ∃ a, initial y o = .mk .fp32 a) (o : Nat) :
    ∃ a, wordSpec M e words y N B initial y o = .mk .fp32 a := by
  unfold wordSpec wordReference
  split
  · exact initialTag o
  · dsimp only
    split
    · exact ⟨_, rfl⟩
    · exact initialTag o

#specNonCircular wordSpec avoiding [VeriTile.Bench.Repairs.ReluTritonKernel.kernel,runPrefix,runGrid,FP.Structural.exec,reference,resultValue]
#axiomsClean withInputWords_encoded
#axiomsClean separated_output_preserved
#axiomsClean reference_refines
#axiomsClean wordSpec_frame
#axiomsClean wordSpec_written_fp32
#axiomsClean wordSpec_output_tag
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.IntegerWords
