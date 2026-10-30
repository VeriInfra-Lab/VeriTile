import bench.tritonbench_g.triton_argmax.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- The complete quoted dimension program stores precisely the current-input
index used by Native.dimension's byte grid, with the exact quoted store frame.
This supplies the source-to-byte-graph link for the returned Tensor theorem. -/
theorem dimension (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (c : DimConfig) (g : Native.DimPrograms c) (base : Nat) (index64 : Bool)
    (nonempty : 0<c.extent) (innerPositive : 0<c.inner) :
    ∃ t, FP.Structural.exec (Typed.backend rest)
      (TypedSource.dimension X O c.outer c.extent c.inner 128 128 base index64) s=some t ∧
      (∀ i : Fin 128,((s.pids 0+base)/c.inner)*128+i.val<c.outer →
        t.mem (Region.cast O) ((((s.pids 0+base)/c.inner)*128+i.val)*c.inner+(s.pids 0+base)%c.inner)=
          .mk .int (Int.ofNat (Native.dimIndex g (fun address => (s.mem X address).read .real)
            (((s.pids 0+base)/c.inner)*128+i.val) ((s.pids 0+base)%c.inner)))) ∧
      (∀ region offset,(region≠Region.cast O ∨
        ∀ i : Fin 128,((s.pids 0+base)/c.inner)*128+i.val<c.outer →
          offset≠((((s.pids 0+base)/c.inner)*128+i.val)*c.inner+(s.pids 0+base)%c.inner)) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t,run,values,frame⟩ := Typed.source_reference rest s X O c.outer c.extent c.inner 128 128 base index64
    (by decide) nonempty innerPositive
  refine ⟨t,run,?_,frame⟩
  intro i active
  rw [values i active,Native.dim_index,dimension_reference c _ _ _ active nonempty
    (fun tile => g.numeric _ _ tile ▸ (g.sites _ _ tile).numericComplete)]
  rfl

#axiomsClean dimension
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedNative
