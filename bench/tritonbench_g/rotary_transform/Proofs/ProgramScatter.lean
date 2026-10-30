import bench.tritonbench_g.rotary_transform.Proofs.ProgramModel

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]

theorem data_injective (s : Shape) (heads : 0<s.heads) (u : State α)
    (a b c d : Nat) (hc : c<s.width) (hd : d<s.width)
    (eq : dataOffset s u a c=dataOffset s u b d) : a=b ∧ c=d := by
  have width : 0<s.width := by omega
  have stride : 0<s.heads*s.width := Nat.mul_pos heads width
  have covers : s.width≤s.heads*s.width := by nlinarith
  have hc' := lt_of_lt_of_le hc covers
  have hd' := lt_of_lt_of_le hd covers
  have eq' : row s u a*(s.heads*s.width)+c=row s u b*(s.heads*s.width)+d :=
    Nat.add_left_cancel eq
  have div (k z : Nat) (hz : z<s.heads*s.width) :
      (k*(s.heads*s.width)+z)/(s.heads*s.width)=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ stride,Nat.div_eq_of_lt hz]
    simp
  have rows := congrArg (fun n => n/(s.heads*s.width)) eq'
  simp only [div _ _ hc',div _ _ hd'] at rows
  rw [rows] at eq'
  exact ⟨Nat.add_left_cancel rows,Nat.add_left_cancel eq'⟩

theorem active_col_bound (s : Shape) (rotary : 2*s.half≤s.width)
    (u : State α) (lane col : Nat) (inter : Bool) (live : active s u lane col inter) : col<s.width := by
  have bound := live.2
  cases inter <;> simp only [if_true,Bool.false_eq_true,if_false] at bound <;> omega

theorem scatter_readback (s : Shape) (heads : 0<s.heads) (rotary : 2*s.half≤s.width)
    (u current : State α) (columns : Nat) (inter : Bool) (displacement : Nat)
    (value : Nat→Nat→α) (i : TileIndex [s.blockM,columns])
    (live : active s u i.1.val i.2.1.val inter) :
    (scatter s u current columns inter displacement value).mem
      (KernelAlgebra.regionName s.output) (dataOffset s u i.1.val i.2.1.val+displacement)=
        Cell.mk .real (value i.1.val i.2.1.val) := by
  apply masked_readback_active _ _ _ _ current i live
  intro a b ha hb eq
  have ij := data_injective s heads u a.1.val b.1.val a.2.1.val b.2.1.val
    (active_col_bound s rotary u _ _ inter ha) (active_col_bound s rotary u _ _ inter hb)
    (Nat.add_right_cancel eq)
  exact Prod.ext (Fin.ext ij.1) (Prod.ext (Fin.ext ij.2) (Subsingleton.elim _ _))

theorem scatter_frame (s : Shape) (u current : State α) (columns : Nat)
    (inter : Bool) (displacement : Nat) (value : Nat→Nat→α) (r : RegionName) (o : Nat)
    (outside : r≠KernelAlgebra.regionName s.output ∨
      ∀ i : TileIndex [s.blockM,columns],active s u i.1.val i.2.1.val inter →
        o≠dataOffset s u i.1.val i.2.1.val+displacement) :
    (scatter s u current columns inter displacement value).mem r o=current.mem r o := by
  apply State.masked_scatter_frame
  exact outside.imp id (fun h i _ => h i)

@[simp] theorem scatter_pids (s : Shape) (u current : State α) (columns : Nat)
    (inter : Bool) (displacement : Nat) (value : Nat→Nat→α) :
    (scatter s u current columns inter displacement value).pids=current.pids := by
  simp [scatter]

theorem result_other (s : Shape) (p : Ops α) (u : State α) (r : RegionName) (o : Nat)
    (other : r≠KernelAlgebra.regionName s.output) : (result s p u).mem r o=u.mem r o := by
  unfold result
  split
  · rfl
  · split
    · exact scatter_frame _ _ _ _ _ _ _ _ _ (Or.inl other)
    · rw [scatter_frame _ _ _ _ _ _ _ _ _ (Or.inl other),
        scatter_frame _ _ _ _ _ _ _ _ _ (Or.inl other)]

@[simp] theorem result_pids (s : Shape) (p : Ops α) (u : State α) :
    (result s p u).pids=u.pids := by
  unfold result
  split
  · rfl
  · split <;> simp only [scatter_pids]

#axiomsClean data_injective
#axiomsClean scatter_readback
#axiomsClean scatter_frame
#axiomsClean result_other
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
