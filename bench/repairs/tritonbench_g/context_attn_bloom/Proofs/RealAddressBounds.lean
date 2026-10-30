import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.RealReadBinding

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

namespace RealInputAgreement
theorem bounds_transfer (p : Layout) (r : Regions) (extent : RegionBounds) (l : Launch) (a b : Math.Inputs)
    (h : Agree r extent l a b) (localBounds : AddressBounds.Valid p r l b extent) (hBN : 0<p.BN)
    (bounds : RegionBounds) (hb : AddressBounds.Valid p r l b bounds) :
    AddressBounds.Valid p r l a bounds := by
  refine ⟨hb.start,hb.prompt,hb.sequence,hb.request,?_,?_,?_,?_,?_⟩
  · intro i d active dim
    rw [h.sequence_eq] at active
    simpa only [AddressBounds.queryAddress,h.start] using hb.query i d active dim
  · intro c hc j active
    rw [h.count_eq] at hc
    rw [h.limit_eq] at active
    simpa only [AddressBounds.tokenAddress,h.request] using hb.tokens c hc j active
  · intro c hc j d active dim
    rw [h.count_eq] at hc
    rw [h.limit_eq] at active
    have visited : c*p.BN+j.val<Math.count p l b*p.BN := by
      have hj:=j.isLt
      nlinarith
    simpa only [AddressBounds.keyAddress,
      token_eq p r extent l a b h localBounds hBN _ visited active] using hb.key c hc j d active dim
  · intro c hc j d active dim
    rw [h.count_eq] at hc
    rw [h.limit_eq] at active
    have visited : c*p.BN+j.val<Math.count p l b*p.BN := by
      have hj:=j.isLt
      nlinarith
    simpa only [AddressBounds.valueAddress,
      token_eq p r extent l a b h localBounds hBN _ visited active] using hb.value c hc j d active dim
  · intro i d active dim
    rw [h.sequence_eq] at active
    simpa only [AddressBounds.outputAddress,h.start] using hb.output i d active dim
end RealInputAgreement

namespace RealKernelIO

theorem inside_mono {address : Int} {left right : Nat}
    (h : AddressBounds.inside address left) (hle : left≤right) : AddressBounds.inside address right :=
  ⟨h.1,lt_of_lt_of_le h.2 (Int.ofNat_le.mpr hle)⟩

theorem logical_bounds (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState) (bounds : RegionBounds)
    (hb : (io g r extent).InBounds bounds s xs) (idx : GridIndex (Host.grid g.arguments)) :
    AddressBounds.Valid (Host.layout g.arguments) r (Host.launchOf g.arguments idx) (mathInputs r extent xs) bounds := by
  have localBounds := valid.bounds idx
  have capacity := input_capacity g r extent xs s bounds hb
  refine ⟨Nat.lt_of_lt_of_le localBounds.start (capacity r.start (by simp)),
    Nat.lt_of_lt_of_le localBounds.prompt (capacity r.prompt (by simp)),
    Nat.lt_of_lt_of_le localBounds.sequence (capacity r.sequence (by simp)),
    Nat.lt_of_lt_of_le localBounds.request (capacity r.request (by simp)),?_,?_,?_,?_,?_⟩
  · intro i d active dim
    exact inside_mono (localBounds.query i d active dim) (capacity r.query (by simp))
  · intro c hc j active
    exact inside_mono (localBounds.tokens c hc j active) (capacity r.tokens (by simp))
  · intro c hc j d active dim
    exact inside_mono (localBounds.key c hc j d active dim) (capacity r.key (by simp))
  · intro c hc j d active dim
    exact inside_mono (localBounds.value c hc j d active dim) (capacity r.value (by simp))
  · intro i d active dim
    have backed := hb.2.1 (0 : Fin 1) ((KernelIO.coordinate g).symm (idx,(i,d,PUnit.unit)))
    simp only [io,Equiv.apply_symm_apply] at backed
    have bound := backed ⟨active,dim⟩
    have nonnegative := (localBounds.output i d active dim).1
    change (AddressBounds.outputAddress _ _ _ i.val d.val).toNat<bounds r.out at bound
    exact ⟨nonnegative,by omega⟩

theorem actual_bounds (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (bounds : RegionBounds)
    (hb : (io g r extent).InBounds bounds s xs) (idx : GridIndex (Host.grid g.arguments)) :
    AddressBounds.Valid (Host.layout g.arguments) r (Host.launchOf g.arguments idx) (RealSetup.inputs s r) bounds :=
  RealInputAgreement.bounds_transfer _ r extent _ _ _
    (input_agreement g r extent xs valid s hin idx) (valid.bounds idx)
    (Host.block_positive g.arguments) bounds (logical_bounds g r extent xs valid s bounds hb idx)

#axiomsClean actual_bounds
end RealKernelIO
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001
