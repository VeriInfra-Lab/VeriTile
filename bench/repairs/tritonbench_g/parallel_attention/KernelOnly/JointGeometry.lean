import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointBacking

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open TensorBinding
noncomputable section
set_option maxHeartbeats 500000
variable {shape : Shape} {read : Memory} {source : State Word}

theorem output_address_bound (shape : Shape) (L : Nat) (p : Program shape L)
    (i : TileIndex [L,shape.BV]) (active : (outputRectangle shape p.tile L).active i) :
    (outputRectangle shape p.tile L).address i<shape.B*shape.H*shape.T*shape.V := by
  have bound := dense_bound shape.T shape.V (shape.B*shape.H) p.head.val
    (p.chunk.val*L+i.1.val) (p.value.val*shape.BV+i.2.1.val) p.head.isLt active.1 active.2
  convert bound using 1 <;>
    simp only [outputRectangle,Program.tile,Rectangle.address,dense,Nat.mul_zero,Nat.add_zero] <;> ring

theorem query_address_bound (shape : Shape) (p : Program shape 64)
    (i : TileIndex [64,shape.BK]) (active : (queryRectangle shape p.tile).active i) :
    (queryRectangle shape p.tile).address i<blocks shape.V shape.BV*shape.B*shape.H*shape.T*shape.K := by
  have head := p.head.isLt
  have shard := p.value.isLt
  have combined : p.head.val+shape.B*shape.H*p.value.val<blocks shape.V shape.BV*(shape.B*shape.H) := by
    nlinarith
  have feature : i.2.1.val<shape.K := by simpa [queryRectangle,Program.tile,Rectangle.active] using active.2
  have bound := dense_bound shape.T shape.K (blocks shape.V shape.BV*(shape.B*shape.H))
    (p.head.val+shape.B*shape.H*p.value.val) (p.chunk.val*64+i.1.val) i.2.1.val combined active.1 feature
  convert bound using 1 <;>
    (try simp only [queryRectangle,Program.tile,Rectangle.address,dense,Nat.zero_mul,Nat.zero_add]) <;> ring

theorem normalizer_address_bound (shape : Shape) (p : Program shape 128) (r : Fin 128)
    (active : p.chunk.val*128+r.val<shape.T) :
    normalizerAddress shape p.tile r.val<shape.B*shape.H*shape.T := by
  have head := p.head.isLt
  simp only [normalizerAddress,Program.tile,Nat.mul_zero,Nat.add_zero]
  nlinarith

theorem forward_store_inside
    (b : Outputs ForwardPort ForwardPort.region (fun k => forwardShape shape k.region) read source)
    (p : Program shape 128) (k : ForwardPort) (i : Nat) (written : forwardWrites shape p (b.view k).region i) :
    i<(b.view k).tensor.metadata.numel := by
  rw [b.region k] at written
  change i<(b.view k).tensor.metadata.shape.prod
  rw [b.shape k]
  cases k with
  | output =>
    rcases written with h|h
    · rcases h.2 with ⟨lane,active,rfl⟩
      simpa [forwardShape,ForwardPort.region,List.prod_cons,Nat.mul_assoc] using output_address_bound shape 128 p lane active
    · simpa [ForwardPort.region] using h.2.1
  | normalizer =>
    rcases written with h|h
    · simpa [ForwardPort.region,outputRectangle] using h.1
    · rcases h.2.2 with ⟨r,active,rfl⟩
      simpa [forwardShape,ForwardPort.region,List.prod_cons,Nat.mul_assoc] using normalizer_address_bound shape p r active

theorem backward_store_inside
    (b : Outputs BackwardPort BackwardPort.region (fun k => backwardShape shape k.region) read source)
    (p : Program shape 64) (k : BackwardPort) (i : Nat) (written : backwardWrites shape p (b.view k).region i) :
    i<(b.view k).tensor.metadata.numel := by
  rw [b.region k] at written
  change i<(b.view k).tensor.metadata.shape.prod
  rw [b.shape k]
  cases k with
  | query =>
    rcases written with h|h|h
    · rcases h.2 with ⟨lane,active,rfl⟩
      simpa [backwardShape,BackwardPort.region,List.prod_cons,Nat.mul_assoc] using query_address_bound shape p lane active
    · simpa [BackwardPort.region,keyRectangle,queryRectangle] using h.1
    · simpa [BackwardPort.region,valueRectangle,outputRectangle] using h.1
  | key =>
    rcases written with h|h|h
    · simpa [BackwardPort.region,queryRectangle] using h.1
    · rcases h.2 with ⟨lane,active,rfl⟩
      simpa [backwardShape,BackwardPort.region,List.prod_cons,Nat.mul_assoc] using query_address_bound shape p lane active
    · simpa [BackwardPort.region,valueRectangle,outputRectangle] using h.1
  | value =>
    rcases written with h|h|h
    · simpa [BackwardPort.region,queryRectangle] using h.1
    · simpa [BackwardPort.region,keyRectangle,queryRectangle] using h.1
    · rcases h.2 with ⟨lane,active,rfl⟩
      simpa [backwardShape,BackwardPort.region,List.prod_cons,Nat.mul_assoc] using output_address_bound shape 64 p lane active

/-- No two programs or ports write the same physical word. This lifts
the existing source-grid ownership proof through actual Tensor addresses. -/
theorem forward_physical_unique
    (b : Outputs ForwardPort ForwardPort.region (fun k => forwardShape shape k.region) read source)
    (p q : Program shape 128) (k l : ForwardPort) (i j : Nat)
    (wp : forwardWrites shape p (b.view k).region i) (wq : forwardWrites shape q (b.view l).region j)
    (same : address (b.view k) i=address (b.view l) j) : p=q ∧ k=l ∧ i=j := by
  have slot : (⟨k,i⟩ : Slot ForwardPort)=⟨l,j⟩ := b.address_unique _ _
    (forward_store_inside b p k i wp) (forward_store_inside b q l j wq) same
  have ports : k=l := congrArg Slot.port slot
  have indices : i=j := congrArg Slot.index slot
  subst l
  subst j
  exact ⟨forward_write_unique shape p q _ _ wp wq,rfl,rfl⟩

theorem backward_physical_unique
    (b : Outputs BackwardPort BackwardPort.region (fun k => backwardShape shape k.region) read source)
    (p q : Program shape 64) (k l : BackwardPort) (i j : Nat)
    (wp : backwardWrites shape p (b.view k).region i) (wq : backwardWrites shape q (b.view l).region j)
    (same : address (b.view k) i=address (b.view l) j) : p=q ∧ k=l ∧ i=j := by
  have slot : (⟨k,i⟩ : Slot BackwardPort)=⟨l,j⟩ := b.address_unique _ _
    (backward_store_inside b p k i wp) (backward_store_inside b q l j wq) same
  have ports : k=l := congrArg Slot.port slot
  have indices : i=j := congrArg Slot.index slot
  subst l
  subst j
  exact ⟨backward_write_unique shape p q _ _ wp wq,rfl,rfl⟩

#axiomsClean output_address_bound
#axiomsClean query_address_bound
#axiomsClean normalizer_address_bound
#axiomsClean forward_store_inside
#axiomsClean backward_store_inside
#axiomsClean forward_physical_unique
#axiomsClean backward_physical_unique
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking
