import bench.repairs.tritonbench_g.triton_mul2.Correct

namespace VeriTile.Bench.Repairs.TritonMul2.Correct
open VeriTile.Triton VeriTile.Examples
open VeriTile.Bench.TritonBenchG.TritonMul2
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The ordinary candidate retains the original masked contract. -/
def commutedIO (in_ptr0 out_ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2IO in_ptr0 out_ptr n B with
    kernel := commutedSourceKernel in_ptr0 out_ptr n B
    projection := by rfl }

theorem mul2_commuted_correctness (in_ptr0 out_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO in_ptr0 out_ptr n B ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk in_ptr0 out_ptr n B
  · intro bounds s hx hout _
    exact commuted_traceSafe in_ptr0 out_ptr n B bounds s hx hout
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := mul2_kernel_region_run in_ptr0 out_ptr n B s xs hx
    refine ⟨s1, (commuted_exec in_ptr0 out_ptr n B s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

/-- The in-place candidate keeps inp = out = ptr, with the same tail mask. -/
def inplaceCommutedIO (ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2InplaceIO ptr n B with
    kernel := inplaceCommutedSourceKernel ptr n B
    projection := by rfl }

theorem mul2_inplace_commuted_correctness (ptr : RegionName) (n B : Nat) :
    Spec.Real (inplaceCommutedIO ptr n B ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact inplace_commuted_flattenOk ptr n B
  · intro bounds s hx _ _
    exact inplace_commuted_traceSafe ptr n B bounds s hx
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := mul2_inplace_kernel_region_run ptr n B s xs hx
    refine ⟨s1, (inplace_commuted_exec ptr n B s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

#stmtSurfaceSubset mul2_commuted_correctness ⊆
  [Spec.Real, RegionName, commutedIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#stmtSurfaceSubset mul2_inplace_commuted_correctness ⊆
  [Spec.Real, RegionName, inplaceCommutedIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#axiomsClean mul2_commuted_correctness
#axiomsClean mul2_inplace_commuted_correctness
end
end VeriTile.Bench.Repairs.TritonMul2.Correct
