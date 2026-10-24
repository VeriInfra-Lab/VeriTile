import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Contract

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open Contract
noncomputable section
set_option maxHeartbeats 4000000

theorem matrix_numel (t : TensorInfo) (two : t.shape.length=2) : t.numel=VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t := by
  obtain ⟨a,b,h⟩:=List.length_eq_two.mp two
  simp [VeriTile.Torch.TensorMetadata.numel,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns,h]
theorem weights_numel (ok : Admitted t s a indices) : t.w.numel=VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.w*(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w) := by
  have c:=(python_checks_iff t s a indices).mp ok
  rcases c.ranks.2.2.2 with three|four
  · obtain ⟨a,b,d,h⟩:=List.length_eq_three.mp three
    simp [VeriTile.Torch.TensorMetadata.numel,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h]
  · obtain ⟨a,b,d,e,h⟩:=List.length_eq_four.mp four
    have singleton : b=1 := by simpa [h] using c.singleton four
    simp [VeriTile.Torch.TensorMetadata.numel,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h,singleton]

/-- All historical source validity obligations now follow from executable
checks of ordinary original-tensor metadata and signed index observations. -/
theorem valid (t : Inputs) (s : Scalars) (a : Backings) (indices : Nat→Int) (ok : Admitted t s a indices) :
    Valid (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) indices := by
  have c:=(python_checks_iff t s a indices).mp ok
  have hp:=VeriTile.Bench.Repairs.LoraExpandGemv.Contract.power_profile ⟨VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w,by have h:=c.shapes.2.1;omega⟩ c.shapes.1
  have xl:=matrix_numel t.x c.ranks.1
  have yl:=matrix_numel t.y c.ranks.2.1
  have wl:=weights_numel ok
  constructor
  · exact c.shapes.1
  · exact ⟨hp.1,hp.2.1⟩
  · intro he
    exact hp.2.2 (by simpa [Geometry.even,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry,Geometry.block] using he)
  · exact Or.inl c.shapes.2.2.2.2.2
  · exact Or.inl rfl
  · intro b hb;simpa [VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry] using hb
  · intro b hb;simpa [logicalIndex,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry] using c.metadata b hb
  · intro b k hb hk
    change b*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w+k*1<t.x.numel
    rw [xl,c.shapes.2.2.1]
    have br : b<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.x := hb.trans_le c.shapes.2.2.2.1
    change k<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w at hk
    nlinarith
  · intro slot n k hs hn hk
    change slot*(VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w)+n*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w+k*1<t.w.numel
    rw [wl]
    change slot<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.w at hs
    change n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w at hn
    change k<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w at hk
    have inner : n*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w+k<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w := by nlinarith
    have bound:=Nat.mul_le_mul_right (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rank t.w) (Nat.succ_le_of_lt hs)
    rw [Nat.succ_mul] at bound
    omega
  · intro b n hb hn
    change b*VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y+n<t.y.numel
    rw [yl]
    have br : b<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.rows t.y := hb.trans_le c.shapes.2.2.2.2.1
    have nc : n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.columns t.y := hn.trans_le c.shapes.2.2.2.2.2
    nlinarith
  · exact ⟨by have h:=c.limits.1;change t.indices.numel<2^31;omega,c.limits.2.2.1⟩
  · change VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w+VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w<2^31
    have h:=c.limits.2.2.2.2.2.2.1
    omega
  · exact c.limits.2.1
  · simpa [VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry,Geometry.block,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.block] using c.limits.2.2.2.2.2.2.2.1
  · simpa [VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry,Geometry.block,VeriTile.Bench.Repairs.LoraExpandGemv.Contract.block] using c.limits.2.2.2.2.2.2.2.2.1
  · simpa [VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry] using c.limits.2.2.2.2.2.2.2.2.2

#axiomsClean valid
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
