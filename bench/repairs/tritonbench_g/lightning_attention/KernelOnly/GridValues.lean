import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.GridRead
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Inputs
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem grid_forward_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (pid : PID) (hp : ValidPID .forward d pid) (n : Nat) (hn : n<d.NT)
    (i : TileIndex [64,d.BM]) (ht : n*64+i.1.val<d.N) :
    gridMemory M D .forward d (grid .forward d) current "output" (fwdOffset d.parameters pid n i)=
      .mk .real (cast M (fwdOutput M D d.parameters pid input n (fwdCarry M D d.parameters pid input n) i)) := by
  refine full_grid_transform M D .forward d hc input current hin pid hp "output" _ (fun _ => _) ?_ ?_
  · exact ⟨rfl,n*64+i.1.val,ht,i.2.1.val,i.2.1.isLt,Geometry.fwd_address d.parameters pid n i⟩
  · intro mem hm
    change fwdMemory M D d.parameters pid mem mem "output" _=_
    rw [fwdMemory_read M D d.parameters pid mem mem n i hn ht
      (Geometry.channel_active d hc pid.channel hp.2)]
    rw [fwdOutput_inputs M D d.parameters pid mem input hm n]

theorem grid_intra_query_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (pid : PID) (hp : ValidPID .intra d pid) (i : TileIndex [64,d.D]) (ht : pid.channel*64+i.1.val<d.N) :
    gridMemory M D .intra d (grid .intra d) current "queryGradient"
      (pid.head*d.N*d.D+(pid.channel*64+i.1.val)*d.D+i.2.1.val)=
      .mk .real (cast M (intraDQ M D d.parameters pid.head pid.channel input i)) := by
  refine full_grid_transform M D .intra d hc input current hin pid hp "queryGradient" _ (fun _ => _) ?_ ?_
  · apply (Or.inl)
    refine ⟨rfl,i.1.val,?_,ht,i.2.1.val,?_,?_⟩
    · exact i.1.isLt
    · exact i.2.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change intraMemory M D d.parameters pid.head pid.channel mem "queryGradient" _=_
    refine (intra_query_read M D d.parameters pid.head pid.channel mem i ht).trans ?_
    rw [(intra_values_inputs M D d.parameters pid.head pid.channel mem input hm).1]

theorem grid_intra_key_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (pid : PID) (hp : ValidPID .intra d pid) (i : TileIndex [d.D,64]) (ht : pid.channel*64+i.2.1.val<d.N) :
    gridMemory M D .intra d (grid .intra d) current "keyGradient"
      (pid.head*d.N*d.D+(pid.channel*64+i.2.1.val)*d.D+i.1.val)=
      .mk .real (cast M (intraDKT M D d.parameters pid.head pid.channel input i)) := by
  refine full_grid_transform M D .intra d hc input current hin pid hp "keyGradient" _ (fun _ => _) ?_ ?_
  · apply (fun h => Or.inr (Or.inl h))
    refine ⟨rfl,i.2.1.val,?_,ht,i.1.val,?_,?_⟩
    · exact i.2.1.isLt
    · exact i.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change intraMemory M D d.parameters pid.head pid.channel mem "keyGradient" _=_
    refine (intra_key_read M D d.parameters pid.head pid.channel mem i ht).trans ?_
    rw [(intra_values_inputs M D d.parameters pid.head pid.channel mem input hm).2.1]

theorem grid_intra_value_read (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (hc : Host.Contract d) (input current : Memory α) (hin : Inputs current input)
    (pid : PID) (hp : ValidPID .intra d pid) (i : TileIndex [64,d.E]) (ht : pid.channel*64+i.1.val<d.N) :
    gridMemory M D .intra d (grid .intra d) current "valueGradient"
      (pid.head*d.N*d.E+(pid.channel*64+i.1.val)*d.E+i.2.1.val)=
      .mk .real (cast M (intraDV M D d.parameters pid.head pid.channel input i)) := by
  refine full_grid_transform M D .intra d hc input current hin pid hp "valueGradient" _ (fun _ => _) ?_ ?_
  · apply (fun h => Or.inr (Or.inr h))
    refine ⟨rfl,i.1.val,?_,ht,i.2.1.val,?_,?_⟩
    · exact i.1.isLt
    · exact i.2.1.isLt
    · simp only [Geometry.dense,Host.Dimensions.parameters]
      ring
  · intro mem hm
    change intraMemory M D d.parameters pid.head pid.channel mem "valueGradient" _=_
    refine (intra_value_read M D d.parameters pid.head pid.channel mem i ht).trans ?_
    rw [(intra_values_inputs M D d.parameters pid.head pid.channel mem input hm).2.2]

#axiomsClean grid_forward_read
#axiomsClean grid_intra_query_read
#axiomsClean grid_intra_key_read
#axiomsClean grid_intra_value_read
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
