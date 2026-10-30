import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeProjection
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Native
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxHeartbeats 1000000

def read32 (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (flat : Fin pl.arena → BitVec 8) (r : Buffer) (hr : r≠.output)
    (off : Nat) (ho : off<extent storage r) : Word :=
  let byteIndex (k : Fin 4) : Fin pl.arena := ⟨pl.base r+4*off+k.val,by
    have hw : width r=4 := by cases r <;> simp_all [width]
    have bound:=pl.bounded r
    rw [hw] at bound
    omega⟩
  BitVec.ofNat 32 ((flat (byteIndex 0)).toNat+256*(flat (byteIndex 1)).toNat+
    65536*(flat (byteIndex 2)).toNat+16777216*(flat (byteIndex 3)).toNat)

theorem word32_decomposition (w : Word) :
    BitVec.ofNat 32
      ((BitVec.ofNat 8 w.toNat).toNat+256*(BitVec.ofNat 8 (w.toNat/256)).toNat+
        65536*(BitVec.ofNat 8 (w.toNat/65536)).toNat+
        16777216*(BitVec.ofNat 8 (w.toNat/16777216)).toNat)=w := by
  apply BitVec.eq_of_toNat_eq
  have hw:=w.isLt
  simp only [BitVec.toNat_ofNat]
  norm_num at hw ⊢
  omega

theorem input_from_current_bytes (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (flat : Fin pl.arena → BitVec 8) (s : State Word)
    (binding : WordBinding p storage pl flat s) (inputs : RawInputs p s)
    (r : Buffer) (hr : r≠.output) (off : Nat) (ho : off<extent storage r) :
    s.mem (region p r) off=.mk .real (read32 p storage pl flat r hr off ho) := by
  have hin : region p r∈[p.Q,p.K,p.V,p.QS,p.KS] := by cases r <;> simp_all [region]
  obtain ⟨w,hw⟩:=inputs _ hin off
  let byteIndex (k : Fin 4) : Fin pl.arena := ⟨pl.base r+4*off+k.val,by
    have width4 : width r=4 := by cases r <;> simp_all [width]
    have bound:=pl.bounded r
    rw [width4] at bound
    omega⟩
  have byteAt (k : Fin 4) : flat (byteIndex k)=BitVec.ofNat 8 (w.toNat/256^k.val) := by
    have h:=binding.inputs r hr off ho k.val k.isLt (byteIndex k) rfl
    simpa [hw,rawCell] using h
  rw [hw]
  change Cell.mk .real w=Cell.mk .real (BitVec.ofNat 32
    ((flat (byteIndex 0)).toNat+256*(flat (byteIndex 1)).toNat+
      65536*(flat (byteIndex 2)).toNat+16777216*(flat (byteIndex 3)).toNat))
  rw [byteAt,byteAt,byteAt,byteAt]
  norm_num only [Fin.val_zero,Fin.val_one,Fin.val_ofNat,pow_zero,pow_one,Nat.div_one]
  exact congrArg (Cell.mk .real) (word32_decomposition w).symm

theorem allocation_raw (p : Parameters) (abi : InputABI p) (before : State Word)
    (alloc : Allocation p abi before) (inputs : RawInputs p before) : RawInputs p alloc.state := by
  intro r hr off
  have hf : r≠p.Out ∧ r≠alloc.unusedM.region := by
    have hq:=alloc.fresh.2 abi.query (by simp)
    have hk:=alloc.fresh.2 abi.key (by simp)
    have hv:=alloc.fresh.2 abi.value (by simp)
    have hqs:=alloc.fresh.2 abi.queryScale (by simp)
    have hks:=alloc.fresh.2 abi.keyScale (by simp)
    rw [abi.regions.1] at hq
    rw [abi.regions.2.1] at hk
    rw [abi.regions.2.2.1] at hv
    rw [abi.regions.2.2.2.1] at hqs
    rw [abi.regions.2.2.2.2] at hks
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl
    · exact ⟨Ne.symm hq.2.2.1,Ne.symm hq.2.2.2⟩
    · exact ⟨Ne.symm hk.2.2.1,Ne.symm hk.2.2.2⟩
    · exact ⟨Ne.symm hv.2.2.1,Ne.symm hv.2.2.2⟩
    · exact ⟨Ne.symm hqs.2.2.1,Ne.symm hqs.2.2.2⟩
    · exact ⟨Ne.symm hks.2.2.1,Ne.symm hks.2.2.2⟩
  rw [alloc.frame r off hf.1 hf.2]
  exact inputs r hr off

#axiomsClean word32_decomposition
#axiomsClean input_from_current_bytes
#axiomsClean allocation_raw
#specNonCircular read32 avoiding [FP.Structural.exec,Whole.invoke,outputGraph]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Native
