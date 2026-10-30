import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.NativeBindings
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.NativeSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

/-- The finite logical read footprint covers every actual source load. -/
theorem tile_bounds (c : Host.Call) (idx : GridIndex c.grid) (s : State α)
    (metadata : FPSuccess.Metadata c s) (bounds : RegionName → Nat)
    (reads : ∀ port off, off ∈ NativeArrays.tileReads c idx (snapshot c s) port →
      off < bounds (NativeArrays.ports c)[port].buf)
    (writes : ∀ i, c.active idx i → c.offset idx i < bounds c.regions.out) :
    NativeSourceSafety.Bounds c.layout c.regions
      (FPReference.inputs c.layout c.regions (FPSuccess.atIndex s idx)) bounds := by
  let a := NativeArrays.tileInput c idx (snapshot c s)
  change NativeSourceSafety.Bounds c.layout c.regions a bounds
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact reads (3 : Fin 9) a.batch (by simp [NativeArrays.tileReads,a])
  · exact reads (4 : Fin 9) a.batch (by simp [NativeArrays.tileReads,a])
  · exact reads (5 : Fin 9) a.batch (by simp [NativeArrays.tileReads,a])
  · exact reads (6 : Fin 9) a.batch (by simp [NativeArrays.tileReads,a])
  · intro i d on
    apply reads (0 : Fin 9)
    apply List.mem_filterMap.mpr
    refine ⟨⟨i,d,PUnit.unit⟩,TileShape.mem_allIndices _ _,?_⟩
    change (if (FPReference.row c.layout a i.val : Int)<a.length then some _ else none)=_
    rw [if_pos on]
  · intro token on
    apply reads (7 : Fin 9)
    apply List.mem_map.mpr
    refine ⟨token,?_,rfl⟩
    apply List.mem_range.mpr
    change token<(FPReference.keyEnd c.layout a).toNat
    omega
  · intro token on d
    apply reads (1 : Fin 9)
    apply List.mem_flatMap.mpr
    refine ⟨token,?_,?_⟩
    · apply List.mem_range.mpr
      change token<(FPReference.keyEnd c.layout a).toNat
      omega
    · exact List.mem_map.mpr ⟨d,List.mem_finRange d,rfl⟩
  · intro token on d
    apply reads (2 : Fin 9)
    apply List.mem_flatMap.mpr
    refine ⟨token,?_,?_⟩
    · apply List.mem_range.mpr
      change token<(FPReference.keyEnd c.layout a).toNat
      omega
    · exact List.mem_map.mpr ⟨d,List.mem_finRange d,rfl⟩
  · intro i d on
    let j : TileIndex [c.layout.BM,c.layout.D] := ⟨i,d,PUnit.unit⟩
    have active : FPSuccess.active c.layout c.regions (FPSuccess.atIndex s idx) j := on
    have result := writes j ((FPSuccess.active_matches c s metadata idx j).mp active)
    rw [←FPSuccess.offset_matches c s metadata idx j] at result
    exact result

#axiomsClean tile_bounds
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings
