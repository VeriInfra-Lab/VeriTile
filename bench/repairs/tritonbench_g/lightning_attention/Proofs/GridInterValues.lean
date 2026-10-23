import bench.repairs.tritonbench_g.lightning_attention.Proofs.GridValues
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem grid_inter_query_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (s : Fin 2)
    (i : TileIndex [32,d.D]) (ht : ((n.val*2+s.val)*32)+i.1.val<d.N) :
    gridMemory M D .inter d (grid .inter d) current "queryGradient" (head*d.N*d.D+(((n.val*2+s.val)*32)+i.1.val)*d.D+i.2.1.val)=
      if n.val=0 then current "queryGradient" (head*d.N*d.D+(((n.val*2+s.val)*32)+i.1.val)*d.D+i.2.1.val)
      else .mk .real (cast M (add M (dot D (subG M d.parameters head ((n.val*2+s.val)*32) input) (queryCarry M D d.parameters head input n.val) i)
        ((current "queryGradient" (head*d.N*d.D+(((n.val*2+s.val)*32)+i.1.val)*d.D+i.2.1.val)).read .real))) := by
  let transform := fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M (dot D (subG M d.parameters head ((n.val*2+s.val)*32) input) (queryCarry M D d.parameters head input n.val) i) (cell.read .real)))
  change gridMemory M D .inter d (grid .inter d) current "queryGradient" _=transform (current "queryGradient" _)
  refine full_grid_transform M D .inter d hc input current hin ⟨head,0⟩ ⟨hh,by change 0<1; decide⟩ "queryGradient" _ transform ?_ ?_
  · apply (Or.inl)
    refine ⟨rfl,((n.val*2+s.val)*32)+i.1.val,ht,i.2.1.val,?_,?_⟩
    · exact i.2.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change interMemory M D d.parameters head mem mem "queryGradient" _=_
    refine (Current.inter_query_read M D d.parameters head mem mem n s i ht).trans ?_
    rw [queryCarry_inputs M D d.parameters head mem input hm n.val]
    rw [Current.subG_inputs M d.parameters head _ mem input hm]
    rfl

theorem grid_inter_key_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (s : Fin 2)
    (i : TileIndex [d.D,32]) (ht : (Current.reverseRow d.parameters n.val s.val)+i.2.1.val<d.N) :
    gridMemory M D .inter d (grid .inter d) current "keyGradient" (head*d.N*d.D+((Current.reverseRow d.parameters n.val s.val)+i.2.1.val)*d.D+i.1.val)=
      if n.val=0 then current "keyGradient" (head*d.N*d.D+((Current.reverseRow d.parameters n.val s.val)+i.2.1.val)*d.D+i.1.val)
      else .mk .real (cast M (add M (dot D (keyValueCarry M D d.parameters head input n.val) (subVT M d.parameters head (Current.reverseRow d.parameters n.val s.val) input) i)
        ((current "keyGradient" (head*d.N*d.D+((Current.reverseRow d.parameters n.val s.val)+i.2.1.val)*d.D+i.1.val)).read .real))) := by
  let transform := fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M (dot D (keyValueCarry M D d.parameters head input n.val) (subVT M d.parameters head (Current.reverseRow d.parameters n.val s.val) input) i) (cell.read .real)))
  change gridMemory M D .inter d (grid .inter d) current "keyGradient" _=transform (current "keyGradient" _)
  refine full_grid_transform M D .inter d hc input current hin ⟨head,0⟩ ⟨hh,by change 0<1; decide⟩ "keyGradient" _ transform ?_ ?_
  · apply (fun h => Or.inr (Or.inl h))
    refine ⟨rfl,(Current.reverseRow d.parameters n.val s.val)+i.2.1.val,ht,i.1.val,?_,?_⟩
    · exact i.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change interMemory M D d.parameters head mem mem "keyGradient" _=_
    refine (Current.inter_key_read M D d.parameters rfl head mem mem n s i ht).trans ?_
    rw [keyValueCarry_inputs M D d.parameters head mem input hm n.val]
    rw [Current.subVT_inputs M d.parameters head _ mem input hm]
    rfl

theorem grid_inter_value_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (s : Fin 2)
    (i : TileIndex [32,d.E]) (ht : (Current.reverseRow d.parameters n.val s.val)+i.1.val<d.N) :
    gridMemory M D .inter d (grid .inter d) current "valueGradient" (head*d.N*d.E+((Current.reverseRow d.parameters n.val s.val)+i.1.val)*d.E+i.2.1.val)=
      if n.val=0 then current "valueGradient" (head*d.N*d.E+((Current.reverseRow d.parameters n.val s.val)+i.1.val)*d.E+i.2.1.val)
      else .mk .real (cast M (add M (dot D (subK M d.parameters head (Current.reverseRow d.parameters n.val s.val) input) (keyValueCarry M D d.parameters head input n.val) i)
        ((current "valueGradient" (head*d.N*d.E+((Current.reverseRow d.parameters n.val s.val)+i.1.val)*d.E+i.2.1.val)).read .real))) := by
  let transform := fun cell : Cell α => if n.val=0 then cell else .mk .real (cast M (add M (dot D (subK M d.parameters head (Current.reverseRow d.parameters n.val s.val) input) (keyValueCarry M D d.parameters head input n.val) i) (cell.read .real)))
  change gridMemory M D .inter d (grid .inter d) current "valueGradient" _=transform (current "valueGradient" _)
  refine full_grid_transform M D .inter d hc input current hin ⟨head,0⟩ ⟨hh,by change 0<1; decide⟩ "valueGradient" _ transform ?_ ?_
  · apply (fun h => Or.inr (Or.inr h))
    refine ⟨rfl,(Current.reverseRow d.parameters n.val s.val)+i.1.val,ht,i.2.1.val,?_,?_⟩
    · exact i.2.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change interMemory M D d.parameters head mem mem "valueGradient" _=_
    refine (Current.inter_value_read M D d.parameters rfl head mem mem n s i ht).trans ?_
    rw [keyValueCarry_inputs M D d.parameters head mem input hm n.val]
    rw [Current.subK_inputs M d.parameters head _ mem input hm]
    rfl

#axiomsClean grid_inter_query_read
#axiomsClean grid_inter_key_read
#axiomsClean grid_inter_value_read
end
end VeriTile.Bench.Repairs.LightningAttention
