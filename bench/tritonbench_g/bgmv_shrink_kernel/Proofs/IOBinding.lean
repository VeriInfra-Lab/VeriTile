import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.IOGeometry
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.IOValues

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOBinding
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
set_option maxHeartbeats 1000000
variable {α γ : Type} [Inhabited α]
variable (M : Algebra α) (P : Primitives α) (h : Call)
    (indices : Nat → Int) (scale : α) (orders : Orders h) (observe : α → γ)

theorem float_roundtrip (d : FloatDType) (v : Value α d.toTileDType) : ofFloat d (toFloat d v) = v := by
  cases d <;> rfl

theorem map_id (d : TileDType) (v : Value α d) : mapValue id d v = v := by
  cases d <;> rfl

local notation "io" => IO.programIO M P h indices scale orders observe

/-- A typed patch updates only the declared Y coordinates. -/
theorem patch_other (vals : (io).Inputs) (current : (io).RawOutputs) (base : State α)
    (r : RegionName) (offset : Nat) (outside : r ≠ "Y") :
    ((io).patchOutputs vals current base).mem r offset = base.mem r offset := by
  apply (io).toMemoryIO.patchOutputs_frame vals current base
  constructor
  · intro o j _
    have ho : o = 0 := Fin.eq_zero o
    subst o
    exact Or.inl outside
  · intro k
    exact Fin.elim0 k

def currentOutput (current : (io).RawOutputs) (offset : Nat) : α :=
  if hb : offset / h.columns < h.batches then if hn : offset % h.columns < h.N then
    toFloat h.outputType (current ⟨0,(⟨offset/h.columns,hb⟩,⟨offset%h.columns,hn⟩,PUnit.unit)⟩)
  else default else default

def currentData (vals : (io).Inputs) (current : (io).RawOutputs) (base : State α) : Data α :=
  { IOValues.data h indices scale vals with
    output := currentOutput M P h indices scale orders observe current
    padding := base.undef }

theorem current_value (current : (io).RawOutputs) (b : Fin h.batches) (n : Fin h.N) :
    currentOutput M P h indices scale orders observe current (outputOffset h b.val n.val) =
      toFloat h.outputType (current ⟨0,(b,n,PUnit.unit)⟩) := by
  have bound : n.val < h.columns := n.isLt.trans_le h.outputCovered
  have positive : 0 < h.columns := Nat.zero_lt_of_lt bound
  have row : (b.val*h.columns+n.val)/h.columns = b.val := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt bound,Nat.zero_add]
  have col : (b.val*h.columns+n.val)%h.columns = n.val := by simp [Nat.mod_eq_of_lt bound]
  simp only [currentOutput,outputOffset,row,col,dif_pos b.isLt,dif_pos n.isLt]
  apply congrArg (fun j : TileIndex [h.batches,h.N] => toFloat h.outputType (current ⟨0,j⟩))
  exact Prod.ext (Fin.ext row) (Prod.ext (Fin.ext col) rfl)

theorem patch_index (vals : (io).Inputs) (domain : IO.domain h indices vals)
    (base : State α) (binding : (io).InputCells vals base) (current : (io).RawOutputs) (b : Fin h.batches) :
    ((io).patchOutputs vals current base).mem "Indices" b.val = .mk .int (indices b.val) := by
  rw [patch_other M P h indices scale orders observe vals current base "Indices" b.val (by decide)]
  have original := binding 0 (b,PUnit.unit) trivial
  simpa only [IO.programIO,IO.inputs,mapValue,domain.1 b] using original

theorem patch_input (vals : (io).Inputs) (base : State α) (binding : (io).InputCells vals base)
    (current : (io).RawOutputs) (b : Fin h.batches) (active : indices b.val ≠ -1) (k : Fin h.K) :
    ((io).patchOutputs vals current base).mem "X" (inputOffset h b.val k.val) =
      .mk (dtype h.storage).toTileDType
        (ofFloat _ ((currentData M P h indices scale orders observe vals current base).input
          (inputOffset h b.val k.val))) := by
  rw [patch_other M P h indices scale orders observe vals current base "X" _ (by decide)]
  have original := binding 1 (b,k,PUnit.unit) active
  change base.mem "X" (inputOffset h b.val k.val) =
    .mk (dtype h.storage).toTileDType (ofFloat _ (IOValues.input h indices vals (inputOffset h b.val k.val)))
  rw [IOValues.input_value]
  simpa only [IO.programIO,IO.inputs,map_id,float_roundtrip] using original

theorem patch_weight (vals : (io).Inputs) (base : State α) (binding : (io).InputCells vals base)
    (current : (io).RawOutputs) (b : Fin h.batches) (active : indices b.val ≠ -1)
    (n : Fin h.N) (k : Fin h.K) :
    ((io).patchOutputs vals current base).mem "W" (weightOffset h (indices b.val).toNat n.val k.val) =
      .mk (dtype h.storage).toTileDType
        (ofFloat _ ((currentData M P h indices scale orders observe vals current base).weights
          (weightOffset h (indices b.val).toNat n.val k.val))) := by
  rw [patch_other M P h indices scale orders observe vals current base "W" _ (by decide)]
  let slot : Fin (IO.slots h indices) := ⟨(indices b.val).toNat,IO.selected_inside h indices b⟩
  have original := binding 2 (slot,n,k,PUnit.unit) ⟨b,active,rfl⟩
  change base.mem "W" (weightOffset h slot.val n.val k.val) =
    .mk (dtype h.storage).toTileDType (ofFloat _ (IOValues.weights h indices vals (weightOffset h slot.val n.val k.val)))
  rw [IOValues.weight_value]
  simpa only [IO.programIO,IO.inputs,map_id,float_roundtrip] using original

theorem patch_output (vals : (io).Inputs) (current : (io).RawOutputs) (base : State α)
    (b : Fin h.batches) (n : Fin h.N) :
    ((io).patchOutputs vals current base).mem "Y" (outputOffset h b.val n.val) =
      .mk h.outputType.toTileDType (ofFloat _
        ((currentData M P h indices scale orders observe vals current base).output (outputOffset h b.val n.val))) := by
  have result := (io).toMemoryIO.patchOutputs_read vals
    (IOGeometry.output_unique M P h indices scale orders observe vals) current base ⟨0,(b,n,PUnit.unit)⟩ trivial
  change ((io).patchOutputs vals current base).mem "Y" (outputOffset h b.val n.val) =
    Cell.mk h.outputType.toTileDType (ofFloat _ (currentOutput M P h indices scale orders observe current (outputOffset h b.val n.val)))
  rw [current_value]
  simpa only [MemoryIO.outputAddress,IO.programIO,IO.outputs,float_roundtrip] using result

#axiomsClean patch_index
#axiomsClean patch_input
#axiomsClean patch_weight
#axiomsClean patch_output
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOBinding
