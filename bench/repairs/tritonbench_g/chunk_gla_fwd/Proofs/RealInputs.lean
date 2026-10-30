import bench.repairs.tritonbench_g.chunk_gla_fwd.IO

/-! Finite logical channels bind exactly the operands of the independent
attention and standalone-consumer formulas. No initial cell-tag restriction
or condition on inactive padding is introduced. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.RealInputs
open VeriTile Triton RealIO
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

theorem extent {n capacity : Nat} (h : ∀ i : Fin n,i.val<capacity) : n≤capacity := by
  by_contra hn
  have impossible := h ⟨capacity,by omega⟩
  change capacity<capacity at impossible
  omega

theorem producer_read (c : Host.Dimensions) (scale : ℝ) (values : ProducerValues c)
    (s : BlockState) (bound : (producer c scale).InputBinding s values)
    (channel : Fin 3) (i : Nat) (hi : i<c.heads*c.T*c.K) :
    extend (values channel) i=s.readMem ((producer c scale).ibuf channel) i := by
  rw [extend,dif_pos hi]
  exact (bound channel ⟨i,hi⟩ trivial).symm

theorem consumer_read (c : Host.Dimensions) (scale : ℝ) (values : ConsumerValues c)
    (s : BlockState) (bound : (consumer c scale).InputBinding s values)
    (channel : Fin 5) (i : Nat) (hi : i<(consumer c scale).iarity channel) :
    extend (values channel) i=s.readMem ((consumer c scale).ibuf channel) i := by
  change i< ![c.heads*c.T*c.K,c.heads*c.T*c.K,Host.outputSize c,
    c.heads*c.NT*c.K*c.V,Host.attentionSize c] channel at hi
  rw [extend,dif_pos hi]
  exact (bound channel ⟨i,hi⟩ trivial).symm

theorem flat_coordinate {heads T width : Nat} (i : Fin (heads*T*width)) :
    i.divNat.divNat.val*T*width+i.divNat.modNat.val*width+i.modNat.val=i.val := by
  change i.val/width/T*T*width+(i.val/width)%T*width+i.val%width=i.val
  rw [← Nat.add_mul,Nat.div_add_mod',Nat.div_add_mod']

theorem packed {heads T K h t k : Nat} (hh : h<heads) (ht : t<T) (hk : k<K) :
    Logical.featureAddress T K h t k<heads*T*K := by
  simpa only [Logical.featureAddress,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
    using Geometry.attention_bound hh ht hk

theorem attention_formula (c : Host.Dimensions) (scale : ℝ)
    (values : ProducerValues c) (s : BlockState)
    (bound : (producer c scale).InputBinding s values)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val=
      Logical.attention c.T c.K c.BT (producerInputs c scale values) scale h.val t.val j.val := by
  unfold Logical.attention
  split
  · rename_i causal
    have column : t.val/c.BT*c.BT+j.val<c.T := by
      have ht := t.isLt
      have div := Nat.div_add_mod' t.val c.BT
      omega
    unfold Logical.contraction
    apply congrArg (fun x=>x*scale)
    apply Finset.sum_congr rfl
    intro k hk
    have hk' := Finset.mem_range.mp hk
    have rowBound := packed h.isLt t.isLt hk'
    have colBound := packed h.isLt column hk'
    simp only [Logical.term,InterGrid.inputs,producerInputs]
    rw [producer_read c scale values s bound (0 : Fin 3) _ rowBound,
      producer_read c scale values s bound (1 : Fin 3) _ colBound,
      producer_read c scale values s bound (2 : Fin 3) _ rowBound,
      producer_read c scale values s bound (2 : Fin 3) _ colBound]
    rfl
  · rfl

theorem consumer_formula (c : Host.Dimensions) (scale : ℝ)
    (values : ConsumerValues c) (s : BlockState)
    (bound : (consumer c scale).InputBinding s values)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    KernelReal.consumerFormula c (LocalReal.outputInputs OutputGrid.regions s) scale h t v=
      KernelReal.consumerFormula c (consumerInputs c scale values) scale h t v := by
  unfold KernelReal.consumerFormula
  apply congrArg₂ (·+·)
  · apply Finset.sum_congr rfl
    intro k hk
    have hk' := Finset.mem_range.mp hk
    have feature := packed h.isLt t.isLt hk'
    have history : Geometry.historyAddress c.NT c.K c.V h.val (t.val/c.BT) k v.val<
        c.heads*c.NT*c.K*c.V :=
      Geometry.history_bound h.isLt (Geometry.owner_lt c.bt_pos t.isLt) hk' v.isLt
    simp only [LocalReal.outputInputs,OutputGrid.regions,consumerInputs]
    rw [consumer_read c scale values s bound (0 : Fin 5) _ feature,
      consumer_read c scale values s bound (1 : Fin 5) _ feature,
      consumer_read c scale values s bound (3 : Fin 5) _ history]
    rfl
  · apply Finset.sum_congr rfl
    intro j hj
    have causal : j≤t.val%c.BT := by have := Finset.mem_range.mp hj; omega
    have hj' : j<c.BT := lt_of_le_of_lt causal (Nat.mod_lt _ c.bt_pos)
    have column : t.val/c.BT*c.BT+j<c.T := by
      have ht := t.isLt
      have div := Nat.div_add_mod' t.val c.BT
      omega
    have attention : h.val*c.T*c.BT+t.val*c.BT+j<Host.attentionSize c := by
      simpa only [Host.attentionSize,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
        using Geometry.attention_bound h.isLt t.isLt hj'
    have value := Geometry.output_bound h.isLt column v.isLt
    simp only [LocalReal.outputInputs,OutputGrid.regions,consumerInputs]
    rw [consumer_read c scale values s bound (4 : Fin 5) _ attention,
      consumer_read c scale values s bound (2 : Fin 5) _ value]
    rfl

#axiomsClean attention_formula
#axiomsClean consumer_formula
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.RealInputs
