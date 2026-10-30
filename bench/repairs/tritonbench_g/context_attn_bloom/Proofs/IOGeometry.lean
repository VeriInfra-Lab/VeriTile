import bench.repairs.tritonbench_g.context_attn_bloom.IO
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem logical_end (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) :
    FPReference.keyEnd (Host.layout g.arguments) (logicalInputs g r extent xs idx)=
      Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
  simp [FPReference.keyEnd,logicalInputs,Math.limit]

theorem logical_stop (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) :
    (FPReference.blockMask (Host.layout g.arguments) (logicalInputs g r extent xs idx) : Int)*
      FPReference.keyEnd (Host.layout g.arguments) (logicalInputs g r extent xs idx)=
      Bounds.stop (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
  rw [logical_end]
  simp only [FPReference.blockMask,logicalInputs,Bounds.stop,Math.sequence]
  split_ifs <;> simp

theorem logical_page (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) (n j : Nat)
    (active : ((n*(Host.layout g.arguments).BN+j : Nat) : Int)<
      Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs)) :
    FPReference.page (Host.layout g.arguments) (logicalInputs g r extent xs idx)
      (n*(Host.layout g.arguments).BN : Nat) j=
      Math.token (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs)
        (n*(Host.layout g.arguments).BN+j) := by
  simp only [FPReference.page,logical_end]
  have h : ((n*(Host.layout g.arguments).BN : Nat) : Int)+(j:Nat)<
      Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
    simpa only [Nat.cast_add] using active
  rw [if_pos h]
  simp only [logicalInputs,Math.token,metadata,Nat.cast_add,Nat.cast_mul,Int.mul_comm]

theorem source_bounds (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (valid : domain g r extent xs) (idx : GridIndex (Host.grid g.arguments)) :
    IOSourceSafety.Bounds (Host.layout g.arguments) r (logicalInputs g r extent xs idx) extent := by
  have hb := valid.bounds idx
  have countBound (cur : Int)
      (aligned : ∃ n : Nat,cur=(n*(Host.layout g.arguments).BN : Nat))
      (live : cur<(FPReference.blockMask (Host.layout g.arguments) (logicalInputs g r extent xs idx) : Int)*
        FPReference.keyEnd (Host.layout g.arguments) (logicalInputs g r extent xs idx)) :
      ∀ n,cur=(n*(Host.layout g.arguments).BN : Nat) →
        n<Math.count (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
    intro n equal
    apply (Bounds.loop_test _ _ _ n (Host.block_positive g.arguments)).mpr
    simpa only [←logical_stop g r extent xs idx,←equal] using live
  refine ⟨hb.start,hb.prompt,hb.sequence,hb.request,?_,?_,?_,?_,?_⟩
  · intro i d active dim
    exact AddressBounds.inside_toNat (hb.query i d active dim)
  · intro cur nonnegative aligned live j active
    rcases aligned with ⟨n,rfl⟩
    have count := countBound _ ⟨n,rfl⟩ live n rfl
    have active' : ((n*(Host.layout g.arguments).BN+j.val : Nat) : Int)<
        Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
      simpa only [Nat.cast_add,logical_end] using active
    simpa [logicalInputs,AddressBounds.tokenAddress,Nat.cast_add,Nat.cast_mul] using
      AddressBounds.inside_toNat (hb.tokens n count j active')
  · intro cur nonnegative aligned live j d active dim
    rcases aligned with ⟨n,rfl⟩
    have count := countBound _ ⟨n,rfl⟩ live n rfl
    have active' : ((n*(Host.layout g.arguments).BN+j.val : Nat) : Int)<
        Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
      simpa only [Nat.cast_add,logical_end] using active
    rw [logical_page g r extent xs idx n j.val active']
    simpa [logicalInputs,AddressBounds.keyAddress,Nat.cast_mul] using
      AddressBounds.inside_toNat (hb.key n count j d active' dim)
  · intro cur nonnegative aligned live j d active dim
    rcases aligned with ⟨n,rfl⟩
    have count := countBound _ ⟨n,rfl⟩ live n rfl
    have active' : ((n*(Host.layout g.arguments).BN+j.val : Nat) : Int)<
        Math.limit (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) := by
      simpa only [Nat.cast_add,logical_end] using active
    rw [logical_page g r extent xs idx n j.val active']
    simpa [logicalInputs,AddressBounds.valueAddress,Nat.cast_mul] using
      AddressBounds.inside_toNat (hb.value n count j d active' dim)
  · intro i d active dim
    exact AddressBounds.inside_toNat (hb.output i d active dim)

#axiomsClean source_bounds
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
