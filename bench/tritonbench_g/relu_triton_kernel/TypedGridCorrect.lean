import bench.tritonbench_g.relu_triton_kernel.Proofs.HostWrapper
import bench.tritonbench_g.relu_triton_kernel.Proofs.SoftwareFP32

/-! Public backend-parametric value contract for the original pid-zero store.
Twelve logical profiles are represented: signed/unsigned 8/16/32/64-bit labels,
bool, fp16, bf16 and fp32. Integer cells are decoded Int/Nat/Bool values, not
physical byte arrays. No fp64, fp8 native-error, host allocation/layout,
finite-width address or native instruction-scheduling coverage is inferred.
-/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedGridCorrect
open VeriTile.Triton FP.Structural GatedGrid TypedValue
noncomputable section

def spec {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (profile : Profile) (x y : RegionName)
    (N B : Nat) (initial : State α) : Memory α :=
  if gridCount N B = 0 then initial.mem
  else reference M cmp16 cmp32 profile.kind x y N B initial.mem

specification relu_typed_grid_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (profile : Profile) (x y : RegionName)
    (N B : Nat) (initial : State α) :
    ∃ final, runGrid (backend M cmp16 cmp32) (source profile x y N B) N B initial = some final ∧
      final.mem = spec M cmp16 cmp32 profile x y N B initial :=
  grid_correct M cmp16 cmp32 profile x y N B initial

/-- A direct source consequence for caller lengths beyond the harness's five
elements: output index1024 is never filled by the original1024-lane launch. -/
theorem caller_tail_untouched {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (profile : Profile) (x y : RegionName)
    (N : Nat) (initial final : State α)
    (h : runGrid (backend M cmp16 cmp32) (source profile x y N 1024) N 1024 initial = some final) :
    final.mem y 1024 = initial.mem y 1024 := by
  obtain ⟨t, ht, hm⟩ := relu_typed_grid_correctness M cmp16 cmp32 profile x y N 1024 initial
  rw [h] at ht
  have heq : final = t := Option.some.inj ht
  rw [heq, hm]
  unfold spec
  split
  · rfl
  · exact reference_frame M cmp16 cmp32 profile.kind x y N 1024 initial.mem y 1024
      (Or.inr (Or.inl (Nat.le_refl _)))

#specNonCircular spec avoiding [source, materialized, runPrefix, runGrid, FP.Structural.exec]
#axiomsClean relu_typed_grid_correctness
#axiomsClean caller_tail_untouched
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedGridCorrect
