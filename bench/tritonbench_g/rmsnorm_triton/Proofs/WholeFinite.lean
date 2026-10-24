import bench.tritonbench_g.rmsnorm_triton.Proofs.WholeIndex

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Every padded column occurs once; active columns cannot be duplicated by
the chunk/lane enumeration. -/
theorem columns_nodup (p : Parameters) : (columns p).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro t _
    exact (List.nodup_range (n := 1024)).map (fun j k h => by omega)
  · apply (List.nodup_range (n := trips p)).imp
    intro t s different value ht hs
    rcases List.mem_map.mp ht with ⟨i,hi,ei⟩
    rcases List.mem_map.mp hs with ⟨j,hj,ej⟩
    have hib := List.mem_range.mp hi
    have hjb := List.mem_range.mp hj
    apply different
    omega

theorem programGrid_nodup (p : Parameters) : (programGrid p).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro b _
    exact (List.nodup_range (n := p.rows)).map (fun j k h => (Prod.mk.inj h).2)
  · apply (List.nodup_range (n := p.batches)).imp
    intro b c different value hb hc
    rcases List.mem_map.mp hb with ⟨i,_,rfl⟩
    rcases List.mem_map.mp hc with ⟨j,_,equal⟩
    exact different (Prod.mk.inj equal).1.symm

theorem output_bytes_unique (p : Parameters) (unique : OutputUnique p) (base : Nat)
    (b m k b' m' k' lane lane' : Nat)
    (hb : b < p.batches) (hm : m < p.rows) (hk : k < p.width)
    (hb' : b' < p.batches) (hm' : m' < p.rows) (hk' : k' < p.width)
    (hl : lane < 4) (hl' : lane' < 4)
    (equal : base+4*outputOffset p b m k+lane = base+4*outputOffset p b' m' k'+lane') :
    b=b' ∧ m=m' ∧ k=k' ∧ lane=lane' := by
  have offsets : outputOffset p b m k = outputOffset p b' m' k' := by omega
  have same := unique b m k b' m' k' hb hm hk hb' hm' hk' offsets
  exact ⟨same.1,same.2.1,same.2.2,by omega⟩

/-- Machine statements exported with the public outcome theorem. -/
def IndexCorrespondence (p : Parameters) (d : Domain p)
    (f : FiniteAdmission p d preexisting) : Prop :=
  p.width ≤ 2147482624 ∧
  (∀ t, t < trips p →
    (loopBits t).toInt = (t*1024 : Nat) ∧
    (loopBits t+i32 1024).toInt = ((t+1)*1024 : Nat) ∧
    ((loopBits t).slt (i32 p.width) = Bool.true ↔ t*1024 < p.width) ∧
    (loopBits (trips p)).slt (i32 p.width) = Bool.false) ∧
  (∀ b m t j, b < p.batches → m < p.rows → t < trips p → j < 1024 →
    (∀ n ∈ ScalarIntermediates p b m t j, (i32 n).toInt = (n : Int)) ∧
    (columnBits t j).toInt = (t*1024+j : Nat) ∧
    ((columnBits t j).slt (i32 p.width) = Bool.true ↔ t*1024+j < p.width) ∧
    (inputPointer p f.byteBase b m t j).toNat = f.byteBase p.input.allocation+4*inputOffset p b m (t*1024+j) ∧
    (weightPointer p f.byteBase t j).toNat = f.byteBase p.weights.allocation+4*weightOffset p (t*1024+j) ∧
    (outputPointer p f.byteBase b m t j).toNat = f.byteBase p.output.allocation+4*outputOffset p b m (t*1024+j)) ∧
  (∀ b m k, b < p.batches → m < p.rows → k < p.width →
    inputOffset p b m k < d.capacity p.input.allocation ∧
    weightOffset p k < d.capacity p.weights.allocation ∧
    outputOffset p b m k < d.capacity p.output.allocation ∧
    (∀ t ∈ [p.input,p.weights,p.output], ∀ offset, offset < d.capacity t.allocation →
      f.byteBase t.allocation+4*offset+4 ≤ 2^64 ∧ 4 ∣ f.byteBase t.allocation+4*offset) ∧
    (∀ a byte lane, preexisting a → byte < 4*d.capacity a → lane < 4 →
      f.byteBase p.output.allocation+4*outputOffset p b m k+lane ≠ f.byteBase a+byte)) ∧
  (columns p).Nodup ∧ (programGrid p).Nodup

theorem index_correspondence (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting) :
    IndexCorrespondence p d f := by
  refine ⟨(loop_bound_iff p).mp f.loopEnd,loop_exact p d f,?_,
    active_accesses p d f,columns_nodup p,programGrid_nodup p⟩
  intro b m t j hb hm ht hj
  have mask := column_mask_exact p d f t j ht hj
  have ptr := pointers_exact p d f b m t j hb hm ht hj
  exact ⟨intermediates_exact p d f b m t j hb hm ht hj,mask.1,mask.2,ptr.1,ptr.2.1,ptr.2.2⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
