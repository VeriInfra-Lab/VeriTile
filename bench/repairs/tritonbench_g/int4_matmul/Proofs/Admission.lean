import bench.repairs.tritonbench_g.int4_matmul.Proofs.Contract

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host HelperHost
set_option maxHeartbeats 2500000

theorem canonical_tensor (t : TensorInfo) (d : DType) (h : TensorContract t d Bool.false) :
    t.descriptor.Canonical d t.descriptor.rows t.descriptor.columns t.descriptor.device := by
  exact ⟨rfl,rfl,h.strides.1,h.strides.2,h.dtype,rfl,h.offset,h.flags.2.1,h.flags.1,h.flags.2.2⟩
theorem packed_tensor (t : TensorInfo) (d : DType) (h : TensorContract t d Bool.true) :
    t.descriptor.PackedLayout d t.descriptor.rows t.descriptor.columns t.descriptor.device := by
  rcases h.strides with dense|transposed
  · exact Or.inl ⟨rfl,rfl,dense.1,dense.2,h.dtype,rfl,h.offset,h.flags.2.1,h.flags.1,h.flags.2.2⟩
  · exact Or.inr ⟨rfl,rfl,transposed.1,transposed.2,h.dtype,rfl,h.offset,h.flags.2.1,h.flags.1,h.flags.2.2⟩

theorem packed_geometry (w s z : TensorInfo) (h : PackedContract w s z) :
    0<packedK w ∧ 0<packedN w ∧ 0<packedGroup w s ∧
    8∣packedK w ∧ 8∣packedN w ∧ packedGroup w s∣packedK w ∧
    packedK w/8=w.descriptor.rows ∧ packedK w/packedGroup w s=s.descriptor.rows := by
  have kp : 0<packedK w := Nat.mul_pos h.weight.positive.1 (by decide)
  have sp:=h.scales.positive.1
  have divisible : s.descriptor.rows∣packedK w := Nat.dvd_of_mod_eq_zero h.geometry.2.1
  have positive : 0<packedGroup w s := Nat.div_pos (Nat.le_of_dvd kp divisible) sp
  have product : packedGroup w s*s.descriptor.rows=packedK w := Nat.div_mul_cancel divisible
  have groups : packedGroup w s∣packedK w := ⟨s.descriptor.rows,product.symm⟩
  have quotient : packedK w/packedGroup w s=s.descriptor.rows := by
    conv_lhs => rw [←product]
    exact Nat.mul_div_cancel_left _ positive
  exact ⟨kp,h.weight.positive.2,positive,⟨w.descriptor.rows,by simp [packedK,Nat.mul_comm]⟩,
    Nat.dvd_of_mod_eq_zero h.geometry.1,groups,by simp [packedK],quotient⟩

def packedShape (w s : TensorInfo) : Shape := ⟨1,packedN w,packedK w,packedGroup w s⟩
theorem packedShape_valid (w s z : TensorInfo) (h : PackedContract w s z) : (packedShape w s).Valid := by
  obtain ⟨kp,np,gp,k8,n8,groups,_,_⟩:=packed_geometry w s z h
  exact ⟨by change 0<1;decide,np,kp,gp,k8,n8,groups⟩
theorem packed_domain (w s z : TensorInfo) (h : PackedContract w s z) :
    PackedDomain (packedShape w s) w.descriptor s.descriptor z.descriptor := by
  have geometry:=packed_geometry w s z h
  have scaleDevice : s.descriptor.device=w.descriptor.device:=h.device s (by simp) w (by simp)
  have zeroDevice : z.descriptor.device=w.descriptor.device:=h.device z (by simp) w (by simp)
  refine ⟨packedShape_valid w s z h,?_,?_,?_,h.independent,h.bounds.1⟩
  · simpa only [packedShape,geometry.2.2.2.2.2.2.1,packedN] using packed_tensor w .int32 h.weight
  · simpa only [packedShape,geometry.2.2.2.2.2.2.2,h.geometry.2.2.1,scaleDevice] using packed_tensor s .fp32 h.scales
  · simpa only [packedShape,geometry.2.2.2.2.2.2.2,h.geometry.2.2.2.1,h.geometry.2.2.2.2,zeroDevice] using packed_tensor z .int32 h.zeros

def arguments (a w s z : TensorInfo) (out : Option TensorInfo) : Arguments :=
  ⟨matrixShape a w s,a.descriptor,w.descriptor,s.descriptor,z.descriptor,out.map TensorInfo.descriptor⟩

theorem matmul_admitted (a w s z : TensorInfo) (g : Int) (out : Option TensorInfo)
    (h : MatmulContract a w s z g out) : Host.Contract (arguments a w s z out) := by
  have geometry:=packed_geometry w s z h.packed
  have scaleDevice : s.descriptor.device=a.descriptor.device:=h.device s (by simp [matrixInputs]) a (by simp [matrixInputs])
  have zeroDevice : z.descriptor.device=a.descriptor.device:=h.device z (by simp [matrixInputs]) a (by simp [matrixInputs])
  have weightDevice : w.descriptor.device=a.descriptor.device:=h.device w (by simp [matrixInputs]) a (by simp [matrixInputs])
  have basic : [a.descriptor.allocation,w.descriptor.allocation,s.descriptor.allocation,z.descriptor.allocation].Nodup := by
    have hh:=h.independent
    simp only [Independent,matrixInputs,List.map_append,List.map_cons,List.map_nil,List.nodup_append] at hh
    exact hh.1
  refine ⟨?_,?_,?_,?_,?_,basic,?_,?_,?_⟩
  · exact ⟨h.input.positive.1,geometry.2.1,geometry.1,geometry.2.2.1,geometry.2.2.2.1,geometry.2.2.2.2.1,geometry.2.2.2.2.2.1⟩
  · simpa only [arguments,matrixShape,h.inner] using canonical_tensor a .fp32 h.input
  · simpa only [arguments,matrixShape,geometry.2.2.2.2.2.2.1,weightDevice,packedN] using packed_tensor w .int32 h.packed.weight
  · simpa only [arguments,matrixShape,geometry.2.2.2.2.2.2.2,h.packed.geometry.2.2.1,scaleDevice] using packed_tensor s .fp32 h.packed.scales
  · simpa only [arguments,matrixShape,geometry.2.2.2.2.2.2.2,h.packed.geometry.2.2.2.1,h.packed.geometry.2.2.2.2,zeroDevice] using packed_tensor z .int32 h.packed.zeros
  · intro provided hp
    cases eq : out with
    | none=>simp [arguments,eq] at hp
    | some result=>
      have ident : result.descriptor=provided:=by simpa [arguments,eq] using hp
      subst provided
      have hc:=h.output result eq
      have dev : result.descriptor.device=a.descriptor.device:=h.device result (by simp [matrixInputs,eq]) a (by simp [matrixInputs,eq])
      refine ⟨?_,?_⟩
      · simpa only [arguments,matrixShape,hc.shape.1,hc.shape.2,dev] using canonical_tensor result .fp32 hc.tensor
      · have hn:=h.independent
        simp only [Independent,matrixInputs,eq,Option.toList_some,List.map_append,List.map_cons,List.map_nil,List.nodup_append] at hn
        simpa [arguments,ne_comm] using hn.2.2
  · constructor
    all_goals first
      | exact h.packed.bounds.1
      | exact h.bounds _ (by simp [formed,arguments,matrixShape])
  · exact h.bounds _ (by simp [formed,arguments,matrixShape])

def quantShape (w : TensorInfo) (group : Int) : Shape := ⟨1,w.descriptor.columns,w.descriptor.rows,group.toNat⟩
theorem quantShape_valid (w : TensorInfo) (g rank : Int) (h : QuantMetadataContract w g rank) : (quantShape w g).Valid := by
  exact ⟨by change 0<1;decide,h.weight.positive.2,h.weight.positive.1,(by have hg:=h.group.1;change 0<g.toNat;omega),
    Nat.dvd_of_mod_eq_zero h.packed.1,Nat.dvd_of_mod_eq_zero h.packed.2,Nat.dvd_of_mod_eq_zero h.group.2⟩

#axiomsClean canonical_tensor
#axiomsClean packed_tensor
#axiomsClean packed_geometry
#axiomsClean packed_domain
#axiomsClean matmul_admitted
#axiomsClean quantShape_valid
end VeriTile.Bench.Repairs.Int4Matmul.Direct
