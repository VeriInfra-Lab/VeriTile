import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.TensorLanes

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Bytes
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership TensorBinding
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (load32 decode)
noncomputable section

theorem region_outside (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (rg : RegionName) (off : Nat) (b : Fin 4) (bound : off<A.extent rg)
    (outside : rg≠"O" ∧ rg≠"M" ∧ (c.finish≠0 ∨ rg≠"L")) :
    ¬∃ idx lane byte,active c lane ∧ Physical.address c A idx lane byte=4*A.addr rg off+b.val := by
  have safety := Contract.safety admission.checked
  rintro ⟨idx,lane,byte,live,eqn⟩
  have hb := b.isLt
  have he := byte.isLt
  unfold Physical.address at eqn
  have address : (Flat.address c A idx lane).2=A.addr rg off := by omega
  have allocated := Flat.output_bound c safety.valid safety.allocated idx lane
  have x := A.decode_addr safety.disjoint (Flat.member A closed allocated) allocated
  have y := A.decode_addr safety.disjoint (Flat.member A closed bound) bound
  change A.decode (Flat.address c A idx lane).2=some (Ownership.address c idx lane) at x
  rw [address,y] at x
  have same := (Option.some.inj x).symm
  cases lane <;> simp only [Ownership.address,Prod.mk.injEq,active] at same live
  · exact outside.2.1 same.1.symm
  · rcases outside.2.2 with finish|other
    · exact finish live
    · exact other same.1.symm
  · exact outside.1 same.1.symm

theorem tensor_denominator_preserved (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (final : RegionName → Nat → Word) (before : Memory)
    (finish : c.finish≠0) (i : Nat) (bound : i<(r .denominator).tensor.metadata.numel) :
    Torch.FP32.logicalValue (fun addr => ⟨load32 (publish c A final before) addr⟩)
      (r .denominator).tensor (a.laws .denominator).tensor i=
    Torch.FP32.logicalValue (fun addr => ⟨load32 before addr⟩)
      (r .denominator).tensor (a.laws .denominator).tensor i := by
  have metadata := output_metadata c A p r a
  have index := contiguous_offset (r .denominator) (a.laws .denominator) metadata.2.2.2.2.2 i bound
  have capacity := (Contract.safety a.checked).allocated.l
  have bound' : i<A.extent "L" := by
    simp only [Torch.TensorMetadata.numel,metadata.2.2.2.1,List.prod_cons,List.prod_nil,Nat.mul_one] at bound
    have b : i<c.Z*c.H*c.rounded := by simpa only [Nat.mul_assoc] using bound
    omega
  rw [all_six_inputs c A p r a _ .denominator i bound,all_six_inputs c A p r a _ .denominator i bound,index]
  apply congrArg (fun word => (⟨word⟩ : Torch.FP32.Word))
  unfold load32
  apply congrArg decode
  funext b
  exact frame c A final before _ (region_outside c A p r a closed "L" i b bound'
    ⟨by decide,by decide,Or.inl finish⟩)

#axiomsClean region_outside
#axiomsClean tensor_denominator_preserved
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Bytes
