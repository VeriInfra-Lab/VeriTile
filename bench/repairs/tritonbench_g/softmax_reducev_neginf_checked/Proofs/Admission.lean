import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Contract
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Domain
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Binding

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked

theorem check_iff (c : Call) : check c=.ok () ↔ MetadataContract c := by
  have same : MetadataContract c ↔ metadataChecks c ∧ (active c → launchChecks c) :=
    ⟨fun h => ⟨h.metadata,h.launch⟩,fun h => ⟨h.1,h.2⟩⟩
  rw [same]
  by_cases a : metadataChecks c <;> by_cases b : active c <;> by_cases d : launchChecks c <;>
    simp [check,a,b,d]

def host (c : Call) : Host.Call :=
  ⟨batch c,heads c,dimensions c,dim c.values 0,dim c.logics 1,c.maximum⟩

/-- Historical content domain, now derived from the actual scalar metadata
checks by checked_content. It is not a public caller safety premise. -/
structure Content (c : Call) (x : Host.Input α) : Prop where
  lengths : ∀ b, b < batch c → 0 < x.counts b ∧ x.counts b ≤ c.maximum ∧
    x.starts b+x.counts b ≤ dim c.logics 1
  pages : ∀ j,Host.Legal (host c) j → ∀ k,k < x.counts j.batch →
    x.pages (pageOffset (host c) x j k) < dim c.values 0

theorem shape_three (m : VeriTile.Torch.TensorMetadata) (rank : m.shape.length=3) :
    m.shape=[m.shape[0]?.getD 0,m.shape[1]?.getD 0,m.shape[2]?.getD 0] := by
  cases hs : m.shape with
  | nil => simp [hs] at rank
  | cons a rest => cases rest with
    | nil => simp [hs] at rank
    | cons b rest => cases rest with
      | nil => simp [hs] at rank
      | cons c rest => cases rest with
        | nil => simp [hs]
        | cons d rest => simp [hs] at rank

theorem checked_admissible (c : Call) (h : MetadataContract c)
    (x : Host.Input α) (data : Content c x) : Admissible (host c) x := by
  rcases h.metadata with ⟨layouts,xrank,vrank,orank,prank,srank,nrank,power,pos,vheads,
    oshape,sshape,pshape,pdtype,mdtype,ordinary,maxwidth,widths⟩
  have xwidth := widths c.logics (by simp [tensors])
  have vwidth := widths c.values (by simp [tensors])
  have owidth := widths c.output (by simp [tensors])
  have pwidth := widths c.locations (by simp [tensors])
  have xshape := TokenSoftmaxLlama.Checked.shape_two c.logics.metadata xrank
  have vshape := shape_three c.values.metadata vrank
  change _ ≤ 2^31-64 at xwidth vwidth owidth pwidth
  unfold VeriTile.Torch.TensorMetadata.numel at xwidth
  rw [xshape] at xwidth
  simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at xwidth
  unfold VeriTile.Torch.TensorMetadata.numel at vwidth
  rw [vshape] at vwidth
  simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at vwidth
  simp only [VeriTile.Torch.TensorMetadata.numel,oshape,List.prod_cons,List.prod_nil,Nat.mul_one] at owidth
  simp only [VeriTile.Torch.TensorMetadata.numel,pshape,List.prod_cons,List.prod_nil,Nat.mul_one] at pwidth
  refine ⟨power,pos,maxwidth,?_,?_,?_,pwidth,data.lengths,data.pages⟩
  · have first : c.logics.metadata.shape.headD 0=c.logics.metadata.shape[0]?.getD 0 := by
      cases c.logics.metadata.shape <;> rfl
    rw [first] at xwidth
    simpa [host,heads,dim] using xwidth
  · change dim c.values 0*(dim c.values 1*dimensions c) ≤ 2^31-64 at vwidth
    rw [vheads] at vwidth
    simpa [host,Nat.mul_assoc] using vwidth
  · simpa [host,Nat.mul_assoc] using owidth

#axiomsClean check_iff
#axiomsClean checked_admissible
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked

structure Extents (c : Call) : Prop where
  logics : c.logics.metadata.numel=heads c*dim c.logics 1
  values : c.values.metadata.numel=dim c.values 0*heads c*dimensions c
  output : c.output.metadata.numel=batch c*heads c*dimensions c
  locations : c.locations.metadata.numel=batch c*c.maximum
  starts : c.starts.metadata.numel=batch c
  lengths : c.lengths.metadata.numel=batch c

theorem checked_extents (c : Call) (h : MetadataContract c) : Extents c := by
  rcases h.metadata with ⟨layouts,xrank,vrank,orank,prank,srank,nrank,power,pos,vheads,
    oshape,sshape,pshape,pdtype,mdtype,ordinary,maxwidth,widths⟩
  have one : ∀ m : VeriTile.Torch.TensorMetadata,m.shape.length=1 → m.numel=m.shape[0]?.getD 0 := by
    intro m rank
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [TokenSoftmaxLlama.Checked.shape_one m rank]
    have first : m.shape.headD 0=m.shape[0]?.getD 0 := by cases m.shape <;> rfl
    simp [first]
  constructor
  · unfold VeriTile.Torch.TensorMetadata.numel
    rw [TokenSoftmaxLlama.Checked.shape_two c.logics.metadata xrank]
    have first : c.logics.metadata.shape.headD 0=c.logics.metadata.shape[0]?.getD 0 := by
      cases c.logics.metadata.shape <;> rfl
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    change c.logics.metadata.shape.headD 0 * c.logics.metadata.shape[1]?.getD 0 =
      c.logics.metadata.shape[0]?.getD 0 * c.logics.metadata.shape[1]?.getD 0
    rw [first]
  · unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_three c.values.metadata vrank]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    change dim c.values 0*(dim c.values 1*dimensions c)=_
    rw [vheads]
    exact (Nat.mul_assoc _ _ _).symm
  · simp [VeriTile.Torch.TensorMetadata.numel,oshape,Nat.mul_assoc]
  · simp [VeriTile.Torch.TensorMetadata.numel,pshape]
  · have same : c.starts.metadata.numel=c.lengths.metadata.numel := by
      simp [VeriTile.Torch.TensorMetadata.numel,sshape]
    exact same.trans (one c.lengths.metadata nrank)
  · exact one c.lengths.metadata nrank

#axiomsClean checked_extents
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
