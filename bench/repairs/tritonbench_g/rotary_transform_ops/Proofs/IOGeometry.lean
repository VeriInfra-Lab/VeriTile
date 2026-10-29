import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.IOFacts

namespace VeriTile.Bench.Repairs.RotaryTransformOps.IOGeometry
open VeriTile.Triton FP.Structural Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem data_input (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (bounds : RegionName → Nat)
    (backed : (io p c cu64 off64 observe).InBounds bounds vals)
    (u : State α) (md : ProgramModel.MetadataBinding c u)
    (batch : u.pids 1<c.shape.batches) (head : u.pids 2<c.shape.heads)
    (lane col : Nat) (active : ProgramModel.active c.shape u lane col Bool.true) :
    ProgramModel.dataOffset c.shape u lane col<bounds "x" := by
  have hr : ProgramModel.row c.shape u lane<c.length (u.pids 1) := by
    simpa only [ProgramModel.active,ProgramModel.length_eq c valid u md batch,
      Bool.true_eq,if_true,Int.ofNat_lt] using active.1
  have hd : col<c.shape.width := lt_of_lt_of_le active.2 valid.rotary
  have inside := Host.address_bound c.shape _ _ col
    (row_bound c valid _ _ batch hr) head hd
  rw [ProgramModel.offset_eq c u md batch]
  exact backed.1.2.1 (0 : Fin 6) (⟨_,inside⟩,()) trivial

theorem data_output (p : Ops α) (c : Config) (valid : c.Valid) (cu64 off64 : Bool)
    (observe : α → α) (vals : Inputs α c.shape) (bounds : RegionName → Nat)
    (backed : (io p c cu64 off64 observe).InBounds bounds vals)
    (u : State α) (md : ProgramModel.MetadataBinding c u)
    (batch : u.pids 1<c.shape.batches) (head : u.pids 2<c.shape.heads)
    (lane col : Nat) (active : ProgramModel.active c.shape u lane col Bool.true) :
    ProgramModel.dataOffset c.shape u lane col<bounds (KernelAlgebra.regionName c.shape.output) := by
  have hr : ProgramModel.row c.shape u lane<c.length (u.pids 1) := by
    simpa only [ProgramModel.active,ProgramModel.length_eq c valid u md batch,
      Bool.true_eq,if_true,Int.ofNat_lt] using active.1
  have hd : col<c.shape.width := lt_of_lt_of_le active.2 valid.rotary
  rw [ProgramModel.offset_eq c u md batch]
  exact backed.2 (0 : Fin 1)
    (⟨_,batch⟩,⟨_,lt_of_lt_of_le hr (valid.lengthBound _ batch)⟩,⟨_,head⟩,⟨_,hd⟩,()) hr

theorem coefficient (c : Config) (u : State α) (md : ProgramModel.MetadataBinding c u)
    (batch : u.pids 1<c.shape.batches) (lane pair : Nat)
    (active : ProgramModel.hasCoefficient c.shape u lane pair) :
    ProgramModel.coefficientOffset c.shape u lane pair<c.shape.coefficientRows*c.shape.half := by
  have hr : ProgramModel.row c.shape u lane+c.shift (u.pids 1)<c.shape.coefficientRows := by
    simpa only [ProgramModel.shift_eq c u md batch,←Int.natCast_add,Int.ofNat_lt] using active.1
  have bound := Nat.mul_le_mul_right c.shape.half (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at bound
  rw [ProgramModel.coefficient_eq c u md batch]
  change (ProgramModel.row c.shape u lane+c.shift (u.pids 1))*c.shape.half+pair<_
  have hp := active.2
  omega

end
end VeriTile.Bench.Repairs.RotaryTransformOps.IOGeometry
