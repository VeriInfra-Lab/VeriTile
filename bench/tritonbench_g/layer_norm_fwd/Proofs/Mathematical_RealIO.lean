import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Safety
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FlattenSyntax
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FlatContract

/-! Complete finite-memory execution, seven-output mathematical IO and full
cell frame. The launch row and actual allocation extents imply trace safety.
Zero undef is an explicit restriction of the generic flatten bridge; the source
still retains all omitted-other loads and no source memory access is changed. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.RealIO
open VeriTile.Triton
noncomputable section

def Implements (kernel : Kernel) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
  ∀ initial : BlockState, initial.pid < M → N ≤ block → Domain.ReadWriteSeparated f r →
    initial.undef = (fun _ _ => 0) → FlatDomain.Extents f r stride M N A.extent →
    ∃ final, exec (A.flattenKernel kernel) (A.flattenState initial) = some final ∧
      FlatContract.Post A f r stride M N eps p initial final ∧
      (∀ region offset, FlatContract.Outside A f r stride M N initial region offset →
        final.mem region offset = (A.flattenState initial).mem region offset)

theorem original_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    Implements (Execution.source f r stride M N eps p block hBlock) f r stride M N eps p block := by
  intro A hd hc initial hpid hN hsep hu hb
  obtain ⟨final, he, hp⟩ := RealExecution.source_correct f r stride M N eps p block hBlock initial hN hsep
  have hs := Safety.source_safe f r stride M N eps p block hBlock initial hN hsep A.extent
    (FlatDomain.Extents.rowBounds f r stride M N initial.pid A.extent hb hpid)
  have hf := A.exec_flatten hd hc (Execution.source f r stride M N eps p block hBlock)
    initial hs (Safety.source_flatten_ok f r stride M N eps p block hBlock) hu
  have hpost := FlatContract.flatten_post A f r stride M N eps p initial final hp
  refine ⟨A.flattenState final, ?_, hpost, ?_⟩
  · rw [hf, he, Option.map_some]
  · exact FlatContract.frame A f r stride M N eps p initial (A.flattenState final) hpost

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.RealIO
