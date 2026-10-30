import bench.repairs.tritonbench_g.lora_expand_gemv.Contract

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Direct
open Contract
noncomputable section
set_option maxHeartbeats 4000000

theorem matrix_numel (t : TensorInfo) (two : t.shape.length=2) : t.numel=rows t*columns t := by
  obtain ⟨a,b,h⟩:=List.length_eq_two.mp two
  simp [VeriTile.Torch.TensorMetadata.numel,rows,columns,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
    VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns,h]
theorem weights_numel (ok : Admitted t s indices) : t.w.numel=rows t.w*(width t.w*rank t.w) := by
  have c:=(python_checks_iff t s indices).mp ok
  rcases c.ranks.2.2.2 with three|four
  · obtain ⟨a,b,d,h⟩:=List.length_eq_three.mp three
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h]
  · obtain ⟨a,b,d,e,h⟩:=List.length_eq_four.mp four
    have singleton : b=1 := by simpa [h] using c.singleton four
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,h,singleton]

/-- All historical source validity obligations now follow from executable
checks of ordinary original-tensor metadata and signed index observations. -/
theorem valid (t : Inputs) (s : Scalars) (indices : Nat→Int) (ok : Admitted t s indices) :
    Valid (geometry t s) indices := by
  have c:=(python_checks_iff t s indices).mp ok
  have hp:=power_profile ⟨rank t.w,by have h:=c.shapes.2.1;omega⟩ c.shapes.1
  have xl:=matrix_numel t.x c.ranks.1
  have yl:=matrix_numel t.y c.ranks.2.1
  have wl:=weights_numel ok
  constructor
  · exact c.shapes.1
  · exact ⟨hp.1,hp.2.1⟩
  · intro he
    exact hp.2.2 (by simpa [Geometry.even,geometry,Geometry.block] using he)
  · exact Or.inl c.shapes.2.2.2.2.2
  · exact Or.inl rfl
  · intro b hb;simpa [geometry] using hb
  · intro b hb;simpa [logicalIndex,geometry] using c.metadata b hb
  · intro b k hb hk
    change b*rank t.w+k*1<t.x.numel
    rw [xl,c.shapes.2.2.1]
    have br : b<rows t.x := hb.trans_le c.shapes.2.2.2.1
    change k<rank t.w at hk
    nlinarith
  · intro slot n k hs hn hk
    change slot*(width t.w*rank t.w)+n*rank t.w+k*1<t.w.numel
    rw [wl]
    change slot<rows t.w at hs
    change n<width t.w at hn
    change k<rank t.w at hk
    have inner : n*rank t.w+k<width t.w*rank t.w := by nlinarith
    have bound:=Nat.mul_le_mul_right (width t.w*rank t.w) (Nat.succ_le_of_lt hs)
    rw [Nat.succ_mul] at bound
    omega
  · intro b n hb hn
    change b*columns t.y+n<t.y.numel
    rw [yl]
    have br : b<rows t.y := hb.trans_le c.shapes.2.2.2.2.1
    have nc : n<columns t.y := hn.trans_le c.shapes.2.2.2.2.2
    nlinarith
  · exact ⟨by have h:=c.limits.1;change t.indices.numel<2^31;omega,c.limits.2.2.1⟩
  · change width t.w+width t.w<2^31
    have h:=c.limits.2.2.2.2.2.2.1
    omega
  · exact c.limits.2.1
  · simpa [geometry,Geometry.block,block] using c.limits.2.2.2.2.2.2.2.1
  · simpa [geometry,Geometry.block,block] using c.limits.2.2.2.2.2.2.2.2.1
  · simpa [geometry] using c.limits.2.2.2.2.2.2.2.2.2

#axiomsClean valid
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Direct
