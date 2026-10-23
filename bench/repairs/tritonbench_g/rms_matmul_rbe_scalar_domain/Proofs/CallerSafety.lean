import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Caller

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem inBacking_of_lt (t : RmsMatmulRbe.Contract.Tensor)
    (ordinary : RmsMatmulRbe.Checked.Ordinary t) (j : Nat) (hj : j<t.metadata.numel) :
    inBacking t j := by
  exact le_trans (Nat.mul_le_mul_left _ (by omega)) ordinary.span

theorem backing_window (t : RmsMatmulRbe.Contract.Tensor) (j : Nat) (hj : inBacking t j) :
    t.base≤address t j ∧ address t j+RmsMatmulRbe.Contract.width t.metadata.dtype≤t.base+t.bytes := by
  dsimp [address,inBacking] at *
  constructor
  · omega
  · nlinarith

theorem backing_apart (a b : RmsMatmulRbe.Contract.Tensor) (j k : Nat)
    (ha : inBacking a j) (hb : inBacking b k) (separate : RmsMatmulRbe.Contract.disjoint a b) :
    Apart (address a j) (RmsMatmulRbe.Contract.width a.metadata.dtype)
      (address b k) (RmsMatmulRbe.Contract.width b.metadata.dtype) := by
  have wa := backing_window a j ha
  have wb := backing_window b k hb
  unfold RmsMatmulRbe.Contract.disjoint RmsMatmulRbe.Contract.overlap at separate
  unfold Apart
  omega

theorem q_count (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) :
    q.metadata.numel=(RmsMatmulRbe.Checked.shape i.original).B*
      (RmsMatmulRbe.Checked.shape i.original).M*(RmsMatmulRbe.Checked.shape i.original).N := by
  simp [VeriTile.Torch.TensorMetadata.numel,hq.shape,Nat.mul_assoc]

theorem caller_output_count (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3) :
    (RmsMatmulRbe.Checked.output i.original q phase).metadata.numel=
      (RmsMatmulRbe.Checked.shape i.original).B*(RmsMatmulRbe.Checked.shape i.original).M*
      (RmsMatmulRbe.Checked.shape i.original).N := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  fin_cases phase <;> simp [RmsMatmulRbe.Checked.output,q_count i q hq,f.kCount,f.vCount]

theorem caller_output_ordinary (i : Contract.Inputs) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3) :
    RmsMatmulRbe.Checked.Ordinary (RmsMatmulRbe.Checked.output i.original q phase) := by
  fin_cases phase
  · exact hq.ordinary
  · exact ordinary i.original.k (by simp)
  · exact ordinary i.original.v (by simp)

theorem caller_output_admitted (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3) :
    RmsMatmulRbe.Contract.TensorAdmitted (RmsMatmulRbe.Checked.output i.original q phase)
      RmsMatmulRbe.Contract.weightTypes := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  fin_cases phase
  · rcases hq.admitted with ⟨ht,hl,hd,hdev,hc,hr,ha⟩
    exact ⟨ht,hl,by simpa [RmsMatmulRbe.Contract.weightTypes] using Or.inl (List.mem_singleton.mp hd),hdev,hc,hr,ha⟩
  · exact f.others i.original.k (by simp [RmsMatmulRbe.Contract.Inputs.others])
  · exact f.others i.original.v (by simp [RmsMatmulRbe.Contract.Inputs.others])

theorem q_dtype (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) :
    q.metadata.dtype=RmsMatmulRbe.Contract.fp16 := List.mem_singleton.mp hq.admitted.2.2.1

theorem caller_output_width (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3) :
    outputWidth (qkvOutputFP8 (useFP8 i) phase)=
      RmsMatmulRbe.Contract.width (RmsMatmulRbe.Checked.output i.original q phase).metadata.dtype := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  have qw := (f.others i.original.qw (by simp [RmsMatmulRbe.Contract.Inputs.others])).2.2.1
  have hk := f.format i.original.k (by simp [RmsMatmulRbe.Contract.Inputs.sameFormat])
  have hv := f.format i.original.v (by simp [RmsMatmulRbe.Contract.Inputs.sameFormat])
  simp only [RmsMatmulRbe.Contract.weightTypes,List.mem_cons,List.not_mem_nil,or_false] at qw
  fin_cases phase <;> rcases qw with hh|hh <;>
    simp [RmsMatmulRbe.Checked.output,qkvOutputFP8,outputWidth,useFP8,hk,hv,hh,q_dtype i q hq,
      RmsMatmulRbe.Contract.width,RmsMatmulRbe.Contract.fp16,RmsMatmulRbe.Contract.int8,RmsMatmulRbe.Contract.fp32]

theorem caller_output_read_apart (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3)
    (t : RmsMatmulRbe.Contract.Tensor) (ht : t∈i.original.reads) :
    RmsMatmulRbe.Contract.disjoint (RmsMatmulRbe.Checked.output i.original q phase) t := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  fin_cases phase
  · exact hq.separate t (List.mem_append_left _ ht)
  · exact (f.outputsDisjoint t ht).1
  · exact (f.outputsDisjoint t ht).2

theorem caller_outputs_apart (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (p r : Fin 3) (hne : p≠r) :
    RmsMatmulRbe.Contract.disjoint (RmsMatmulRbe.Checked.output i.original q p)
      (RmsMatmulRbe.Checked.output i.original q r) := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  have qk := hq.separate i.original.k (by simp)
  have qv := hq.separate i.original.v (by simp)
  have kq := RmsMatmulRbe.Checked.disjoint_symm _ _ qk
  have vq := RmsMatmulRbe.Checked.disjoint_symm _ _ qv
  have vk := RmsMatmulRbe.Checked.disjoint_symm _ _ f.kvDisjoint
  have kv := f.kvDisjoint
  fin_cases p <;> fin_cases r <;> simp_all [RmsMatmulRbe.Checked.output]

theorem x_index_lt (i : Contract.Inputs) (checked : Contract.InputContract i) (b m k : Nat)
    (hb : b<(RmsMatmulRbe.Checked.shape i.original).B)
    (hm : m<(RmsMatmulRbe.Checked.shape i.original).M)
    (hk : k<(RmsMatmulRbe.Checked.shape i.original).K) :
    b*((RmsMatmulRbe.Checked.shape i.original).M*(RmsMatmulRbe.Checked.shape i.original).K)+
      m*(RmsMatmulRbe.Checked.shape i.original).K+k < i.original.x.metadata.numel := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  rw [f.xCount]
  have bound := RmsMatmulRbe.Checked.flat3_lt _ _ _ b m k hb hm hk
  simpa [RmsMatmulRbe.Checked.flat3,Nat.add_mul,Nat.mul_assoc] using bound

theorem weight_index_lt (i : Contract.Inputs) (checked : Contract.InputContract i)
    (phase : Fin 3) (n k : Nat) (hn : n<(RmsMatmulRbe.Checked.shape i.original).N)
    (hk : k<(RmsMatmulRbe.Checked.shape i.original).K) :
    n*(RmsMatmulRbe.Checked.shape i.original).K+k<
      (RmsMatmulRbe.Checked.weight i.original phase).metadata.numel := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  rw [f.wCount]
  nlinarith

theorem rms_index_lt (i : Contract.Inputs) (checked : Contract.InputContract i) (k : Nat)
    (hk : k<(RmsMatmulRbe.Checked.shape i.original).K) : k < i.original.rms.metadata.numel := by
  rw [(RmsMatmulRbe.Checked.facts i.original checked.1).rCount]
  exact hk

#axiomsClean backing_apart
#axiomsClean caller_output_count
#axiomsClean caller_output_width
#axiomsClean caller_outputs_apart
#axiomsClean x_index_lt
#axiomsClean weight_index_lt
#axiomsClean rms_index_lt
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
