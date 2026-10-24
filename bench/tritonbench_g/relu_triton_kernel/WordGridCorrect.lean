import bench.tritonbench_g.relu_triton_kernel.Proofs.WordGrid

/-! Public finite-input-word refinement for the original ReLU typed grid.
The nine profiles are signed/unsigned8/16/32/64 and raw-byte bool. The input
relation is explicitly required, every written output is tagged fp32, and
the untouched tail is relative to prelaunch memory. Logical aliases are
permitted; no native physical-view, allocator, compiler, error or finite-width
address correspondence is claimed. Integer-to-fp32/zero callbacks remain open.
-/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.WordGridCorrect
open VeriTile.Triton FP.Structural GatedGrid TypedValue IntegerWords
noncomputable section

specification relu_word_grid_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool)
    (encoding : Encoding) (words : Nat → Word encoding)
    (x y : RegionName) (N B : Nat) (initial : State α)
    (inputWords : InputWords encoding words x N initial.mem) :
    ∃ final,
      runGrid (backend M cmp16 cmp32) (source encoding.profile x y N B) N B initial = some final ∧
      final.mem = wordSpec M encoding words y N B initial.mem :=
  grid_correct_words M cmp16 cmp32 encoding words x y N B initial inputWords

/-- Any actual returned state satisfies both exact first-block fp32 readback
and the full frame. Its premise is used only to identify the already-proved
successful result, not as a substitute for a success proof. -/
theorem actual_result_and_frame {α : Type} [Inhabited α]
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool)
    (encoding : Encoding) (words : Nat → Word encoding)
    (x y : RegionName) (N B : Nat) (initial final : State α)
    (inputWords : InputWords encoding words x N initial.mem)
    (actual : runGrid (backend M cmp16 cmp32) (source encoding.profile x y N B) N B initial = some final) :
    (∀ o, o < B → o < N →
      final.mem y o = .mk .fp32 (wordValue M encoding (words o))) ∧
    (∀ r o, r ≠ y ∨ B ≤ o ∨ N ≤ o → final.mem r o = initial.mem r o) := by
  obtain ⟨t, ht, hm⟩ := relu_word_grid_correctness M cmp16 cmp32 encoding words x y N B initial inputWords
  rw [actual] at ht
  have same : final = t := Option.some.inj ht
  subst t
  rw [hm]
  exact ⟨fun o hb hn => wordSpec_written_fp32 M encoding words y N B initial.mem o hb hn,
    fun r o outside => wordSpec_frame M encoding words y N B initial.mem r o outside⟩

#axiomsClean relu_word_grid_correctness
#axiomsClean actual_result_and_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.WordGridCorrect
