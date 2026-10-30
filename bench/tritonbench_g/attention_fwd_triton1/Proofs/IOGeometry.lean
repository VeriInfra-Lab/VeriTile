import bench.tritonbench_g.attention_fwd_triton1.IO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

theorem main_coordinates (c : Shape) (off : Nat) (bound : off<c.mainSize) :
    mainHead c off<c.heads ∧ mainRow c off<c.T ∧ mainColumn c off<c.D := by
  have positive : 0<c.headSize := by
    by_contra h
    have zero : c.headSize=0 := by omega
    simp [Shape.mainSize,zero] at bound
  exact ⟨(Nat.div_lt_iff_lt_mul positive).mpr bound,
    (Nat.div_lt_iff_lt_mul c.dim_pos).mpr (Nat.mod_lt off positive),Nat.mod_lt off c.dim_pos⟩

theorem main_offset (c : Shape) (off : Nat) :
    c.mainOffset (mainHead c off) (mainRow c off) (mainColumn c off)=off := by
  have remainder : (off%c.headSize)%c.D=off%c.D := by
    have eq := congrArg (fun n=>n%c.D) (Nat.div_add_mod off c.headSize)
    simpa only [Shape.headSize,Nat.add_mod,Nat.mul_mod,Nat.mod_self,Nat.mul_zero,Nat.zero_mul,Nat.zero_mod,
      Nat.zero_add,Nat.mod_mod] using eq
  unfold Shape.mainOffset mainHead mainRow mainColumn
  rw [←remainder,Nat.add_assoc,Nat.div_add_mod',Nat.div_add_mod']

theorem history_coordinates (c : Shape) (off : Nat) (bound : off<c.historySize) :
    historyHead c off<c.heads ∧ historyChunk c off<c.chunks ∧
      historyRow c off<c.D ∧ historyColumn c off<c.D := by
  have positive : 0<c.historyHead := by
    by_contra h
    have zero : c.historyHead=0 := by omega
    simp [Shape.historySize,zero] at bound
  have residue := Nat.mod_lt off positive
  have square : 0<c.D*c.D := Nat.mul_pos c.dim_pos c.dim_pos
  exact ⟨(Nat.div_lt_iff_lt_mul positive).mpr bound,
    (Nat.div_lt_iff_lt_mul square).mpr (by simpa only [Shape.historyHead,Nat.mul_assoc] using residue),
    Nat.mod_lt _ c.dim_pos,Nat.mod_lt _ c.dim_pos⟩

theorem history_offset (c : Shape) (off : Nat) :
    c.historyOffset (historyHead c off) (historyChunk c off) (historyRow c off) (historyColumn c off)=off := by
  have remainder : (off%c.historyHead)%c.D=off%c.D := by
    have eq := congrArg (fun n=>n%c.D) (Nat.div_add_mod off c.historyHead)
    simpa only [Shape.historyHead,Nat.add_mod,Nat.mul_mod,Nat.mod_self,Nat.mul_zero,Nat.zero_mul,Nat.zero_mod,
      Nat.zero_add,Nat.mod_mod] using eq
  unfold Shape.historyOffset historyHead historyChunk historyRow historyColumn
  rw [←remainder,←Nat.div_div_eq_div_mul,Nat.div_add_mod',Nat.add_assoc,Nat.div_add_mod',Nat.div_add_mod']

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
