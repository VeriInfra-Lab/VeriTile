import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOInputReads

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IODataBinding
open VeriTile Triton FP.Structural
noncomputable section

theorem from_input_cells {F H : Type} (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (cells : (IOLayout.memory F H s c hasBias).InputCells id vals base) :
    IOInputReads.DataCells s c hasBias vals base := by
  intro i j notC active
  have bound:=cells i j active
  fin_cases i
  all_goals simpa [IOLayout.memory,IOLayout.inputs,mapValue] using bound

/-- Arbitrary output patches keep all finite readonly source inputs. The
split-K C seed is treated by the shared per-address trace, not by this fact. -/
theorem patch {F H : Type} (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals base)
    (current : (IOLayout.memory F H s c hasBias).RawOutputs (Sum F H)) :
    IOInputReads.DataCells s c hasBias vals
      ((IOLayout.memory F H s c hasBias).patchOutputs vals current base) := by
  intro i j notC active
  have region : (IOLayout.inputs s)[i].buf≠("C":RegionName) := by
    fin_cases i
    all_goals simp_all [IOLayout.inputs]
  have miss : ¬(IOLayout.memory F H s c hasBias).OutputHit vals (IOLayout.inputs s)[i].buf
      (TileShape.linearIndex (IOLayout.inputs s)[i].shape j) := by
    rintro ⟨⟨o,l⟩,_,same⟩
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact region (congrArg Prod.fst same).symm
  simp only [MemoryIO.patchOutputs,dif_neg miss]
  exact cells i j notC active

#axiomsClean from_input_cells
#axiomsClean patch
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IODataBinding
