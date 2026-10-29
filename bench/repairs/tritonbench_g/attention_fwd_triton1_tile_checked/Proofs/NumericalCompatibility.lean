import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.NumericalReference
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Contract

/-! Compatibility only for the protected legacy Tensor/physical endpoints.
The protected Contract transitively imports the historical numerical reader.
The new public kernel IO and its proof closure do not import this module. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NumericalCompatibility
open AttentionFwdTriton1.Whole001
noncomputable section
set_option autoImplicit false

theorem scalar_same : NumericalReference.SourceFP.scalar=SourceFP.scalar := rfl
theorem output_same : NumericalReference.SourceFP.output=SourceFP.output := rfl
theorem history_same : NumericalReference.SourceFP.history=SourceFP.history := rfl

theorem numerical_iff (native : ISA) (dom : NumericDomain) (c : Shape)
    (xs : Inputs (BitVec 32)) (scale : BitVec 32) (ifcond : Bool) :
    NumericalReference.BitNumerical native dom c xs scale ifcond ↔
      BitNumerical native dom c xs scale ifcond := by
  constructor
  · rintro ⟨laws,admitted⟩
    exact ⟨⟨laws.mulRN,laws.fmaRN⟩,⟨admitted.output,admitted.carry⟩⟩
  · rintro ⟨laws,admitted⟩
    exact ⟨⟨laws.mulRN,laws.fmaRN⟩,⟨admitted.output,admitted.carry⟩⟩

#axiomsClean numerical_iff
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.NumericalCompatibility
