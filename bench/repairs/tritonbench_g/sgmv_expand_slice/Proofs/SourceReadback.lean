import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceSuffix
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.GridStride

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000

theorem written_read {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (c : Coordinates) (s : State α) (acc : Values α .real [32,32])
    (i : TileIndex [32,32]) (hi : active p c i)
    (injective : ∀ i j, active p c i → active p c j → offset p c i = offset p c j → i = j) :
    (written M abi p c s acc).mem p.output (offset p c i) =
      .mk (storageType abi.output).toTileDType
        (ofFloat (storageType abi.output) (outputValues M abi p c s acc i)) :=
  masked_scatter_readback_active p.output (offset p c) _ (active p c) s i hi injective

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (c : Coordinates) (s : State α) (acc : Values α .real [32,32])
    (region : RegionName) (address : Nat)
    (miss : region ≠ p.output ∨ ∀ i, active p c i → address ≠ offset p c i) :
    (written M abi p c s acc).mem region address = s.mem region address := by
  apply State.masked_scatter_frame
  rcases miss with hr | hi
  · exact Or.inl hr
  · exact Or.inr (fun i _ hai => hi i hai)

theorem active_offset (p : KernelArgs) (c : Coordinates) (i : TileIndex [32,32])
    (hi : active p c i) :
    offset p c i = row c i*p.outputRowStride+column p c i*p.outputColumnStride := by
  simp only [offset,if_pos hi.1,if_pos hi.2]

theorem offset_injective (p : KernelArgs) (c : Coordinates) (outputRows outputColumns : Nat)
    (layout : GridStride.Contiguous outputRows outputColumns p.outputRowStride p.outputColumnStride)
    (rows : c.start+c.length ≤ outputRows) (columns : p.slice+p.columns ≤ outputColumns)
    (i j : TileIndex [32,32]) (hi : active p c i) (hj : active p c j)
    (equal : offset p c i = offset p c j) : i = j := by
  rw [active_offset p c i hi,active_offset p c j hj,
    GridStride.coordinate_linear layout (Nat.lt_of_lt_of_le hi.1 rows) (Nat.lt_of_lt_of_le hi.2 columns),
    GridStride.coordinate_linear layout (Nat.lt_of_lt_of_le hj.1 rows) (Nat.lt_of_lt_of_le hj.2 columns)] at equal
  obtain ⟨hr,hc⟩ := linear_injective (Nat.lt_of_lt_of_le hi.2 columns)
    (Nat.lt_of_lt_of_le hj.2 columns) equal
  rcases i with ⟨ir,ic,iu⟩
  rcases j with ⟨jr,jc,ju⟩
  have er : ir = jr := by apply Fin.ext; simp only [row] at hr; omega
  have ec : ic = jc := by apply Fin.ext; simp only [column] at hc; omega
  subst jr
  subst jc
  congr

#axiomsClean written_read
#axiomsClean written_frame
#axiomsClean offset_injective
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
