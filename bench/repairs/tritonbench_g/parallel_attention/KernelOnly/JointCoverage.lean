import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointGeometry

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section
set_option maxHeartbeats 500000
variable {shape : Shape} {read : Memory} {source : State Word}

def forwardFootprint (shape : Shape) (R : RegionName) (i : Nat) : Prop :=
  ∃ p : Program shape 128,forwardWrites shape p R i
def backwardFootprint (shape : Shape) (R : RegionName) (i : Nat) : Prop :=
  ∃ p : Program shape 64,backwardWrites shape p R i

/-- Every source-written address belongs to one of the simultaneous
physical output ports; no source output is omitted by the joint post. -/
theorem forward_store_port
    (b : Outputs ForwardPort ForwardPort.region (fun k => forwardShape shape k.region) read source)
    (p : Program shape 128) (R : RegionName) (i : Nat) (written : forwardWrites shape p R i) :
    ∃ k : ForwardPort,(b.view k).region=R ∧ i<(b.view k).tensor.metadata.numel := by
  rcases written with h|h
  · have eq : (b.view .output).region=R := (b.region .output).trans h.1.symm
    refine ⟨.output,eq,forward_store_inside b p .output i ?_⟩
    simpa only [eq] using (Or.inl h : forwardWrites shape p R i)
  · have eq : (b.view .normalizer).region=R := (b.region .normalizer).trans h.2.1.symm
    refine ⟨.normalizer,eq,forward_store_inside b p .normalizer i ?_⟩
    simpa only [eq] using (Or.inr h : forwardWrites shape p R i)

theorem backward_store_port
    (b : Outputs BackwardPort BackwardPort.region (fun k => backwardShape shape k.region) read source)
    (p : Program shape 64) (R : RegionName) (i : Nat) (written : backwardWrites shape p R i) :
    ∃ k : BackwardPort,(b.view k).region=R ∧ i<(b.view k).tensor.metadata.numel := by
  rcases written with h|h|h
  · have eq : (b.view .query).region=R := (b.region .query).trans h.1.symm
    refine ⟨.query,eq,backward_store_inside b p .query i ?_⟩
    simpa only [eq] using (Or.inl h : backwardWrites shape p R i)
  · have eq : (b.view .key).region=R := (b.region .key).trans h.1.symm
    refine ⟨.key,eq,backward_store_inside b p .key i ?_⟩
    simpa only [eq] using (Or.inr (Or.inl h) : backwardWrites shape p R i)
  · have eq : (b.view .value).region=R := (b.region .value).trans h.1.symm
    refine ⟨.value,eq,backward_store_inside b p .value i ?_⟩
    simpa only [eq] using (Or.inr (Or.inr h) : backwardWrites shape p R i)

theorem forward_all_stores
    (b : Outputs ForwardPort ForwardPort.region (fun k => forwardShape shape k.region) read source)
    (p : Program shape 128) (R : RegionName) (i : Nat) (written : forwardWrites shape p R i) :
    ∃ k : ForwardPort,(b.view k).region=R ∧
      PhysicalWrites b.view (forwardFootprint shape) (address (b.view k) i) := by
  obtain ⟨k,region,inside⟩ := forward_store_port b p R i written
  refine ⟨k,region,⟨k,i⟩,⟨inside,p,?_⟩,rfl⟩
  simpa only [region] using written

theorem backward_all_stores
    (b : Outputs BackwardPort BackwardPort.region (fun k => backwardShape shape k.region) read source)
    (p : Program shape 64) (R : RegionName) (i : Nat) (written : backwardWrites shape p R i) :
    ∃ k : BackwardPort,(b.view k).region=R ∧
      PhysicalWrites b.view (backwardFootprint shape) (address (b.view k) i) := by
  obtain ⟨k,region,inside⟩ := backward_store_port b p R i written
  refine ⟨k,region,⟨k,i⟩,⟨inside,p,?_⟩,rfl⟩
  simpa only [region] using written

theorem no_program_of_empty (L : Nat) (positive : 0<L)
    (empty : shape.B=0 ∨ shape.H=0 ∨ shape.T=0) (p : Program shape L) : False := by
  have member := program_member shape L p
  rw [programs_empty shape L positive empty] at member
  simp at member

theorem forward_empty_post (read : Memory) (views : κ → View) (final : State Word)
    (empty : shape.B=0 ∨ shape.H=0 ∨ shape.T=0) :
    jointPost read views (forwardFootprint shape) final=read := by
  funext a
  apply jointPost_frame
  rintro ⟨slot,⟨_,p,_⟩,_⟩
  exact no_program_of_empty 128 (by decide) empty p

theorem backward_empty_post (read : Memory) (views : κ → View) (final : State Word)
    (empty : shape.B=0 ∨ shape.H=0 ∨ shape.T=0) :
    jointPost read views (backwardFootprint shape) final=read := by
  funext a
  apply jointPost_frame
  rintro ⟨slot,⟨_,p,_⟩,_⟩
  exact no_program_of_empty 64 (by decide) empty p

#axiomsClean forward_store_port
#axiomsClean backward_store_port
#axiomsClean forward_all_stores
#axiomsClean backward_all_stores
#axiomsClean forward_empty_post
#axiomsClean backward_empty_post
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
