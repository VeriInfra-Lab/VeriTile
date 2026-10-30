import bench.tritonbench_g.triton_attention.Spec

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 2000000

theorem position_lt (c : Shape) (i : Index c) : position c i<c.elements := offset_lt c (coordinate c i)

theorem position_index (c : Shape) (o : Nat) (h : o<c.elements) : position c (index c o h)=o := by
  change (o/c.dim/c.rows*c.rows+o/c.dim%c.rows)*c.dim+o%c.dim=o
  rw [Nat.div_add_mod',Nat.div_add_mod']

theorem position_injective (c : Shape) : Function.Injective (position c) := by
  intro i j he
  have column : i.2.2.1=j.2.2.1 := by
    apply Fin.ext
    have eq := congrArg (·%c.dim) he
    simpa only [position,coordinate,offset,Nat.mul_add_mod_self_right,
      Nat.mod_eq_of_lt i.2.2.1.isLt,Nat.mod_eq_of_lt j.2.2.1.isLt] using eq
  have div (r : Nat) (d : Fin c.dim) : (r*c.dim+d.val)/c.dim=r := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ c.dim_pos,Nat.div_eq_of_lt d.isLt,Nat.zero_add]
  have row : i.1.val*c.rows+i.2.1.val=j.1.val*c.rows+j.2.1.val := by
    have eq:=congrArg (·/c.dim) he
    simpa only [position,coordinate,offset,div] using eq
  have rowIndex : i.2.1=j.2.1 := by
    apply Fin.ext
    have eq:=congrArg (·%c.rows) row
    simpa only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt i.2.1.isLt,Nat.mod_eq_of_lt j.2.1.isLt] using eq
  have head : i.1=j.1 := by
    apply Fin.ext
    have eq:=congrArg Fin.val rowIndex
    have positive:=c.rows_pos
    nlinarith
  exact Prod.ext head (Prod.ext rowIndex (Prod.ext column (Subsingleton.elim _ _)))

theorem index_position (c : Shape) (i : Index c) : index c (position c i) (position_lt c i)=i :=
  position_injective c (position_index c _ _)

theorem input_cells {α β γ : Type} (zero : α) (c : Shape) (encode : α→β)
    (observe : β→α) (jobs : List (MaterializedInvocation β γ)) (xs : Inputs α c)
    (mem : Grid.Memory β) (bound : (layout c encode observe jobs).InputCells xs mem)
    (r : RegionName) (member : r∈(["Q","K","V","DO"] : List RegionName))
    (o : Nat) (inside : o<c.elements) :
    mem r o=.mk .fp32 (encode (inputAt zero c xs r o)) := by
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl
  · simpa [layout,inputs,inputOffset,inputAt,inside,position_index,MaterializedProgramIO.InputCells,mapValue] using
      bound (0 : Fin 5) (index c o inside) trivial
  · simpa [layout,inputs,inputOffset,inputAt,inside,position_index,MaterializedProgramIO.InputCells,mapValue] using
      bound (1 : Fin 5) (index c o inside) trivial
  · simpa [layout,inputs,inputOffset,inputAt,inside,position_index,MaterializedProgramIO.InputCells,mapValue] using
      bound (2 : Fin 5) (index c o inside) trivial
  · simpa [layout,inputs,inputOffset,inputAt,inside,position_index,MaterializedProgramIO.InputCells,mapValue] using
      bound (3 : Fin 5) (index c o inside) trivial

theorem input_zero {α β γ : Type} (zero : α) (c : Shape) (encode : α→β)
    (observe : β→α) (jobs : List (MaterializedInvocation β γ)) (xs : Inputs α c)
    (admitted : ZeroDQ zero c xs) (mem : Grid.Memory β)
    (bound : (layout c encode observe jobs).InputCells xs mem) (i : Coord c) :
    mem "DQ" (offset c i)=.mk .fp32 (encode zero) := by
  have cell := bound (4 : Fin 5) (i.1,i.2.1,i.2.2,PUnit.unit) trivial
  change mem "DQ" (offset c i)=.mk .fp32 (encode (xs 4 (i.1,i.2.1,i.2.2,PUnit.unit))) at cell
  rw [admitted] at cell
  exact cell

#axiomsClean input_cells
#axiomsClean input_zero
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
