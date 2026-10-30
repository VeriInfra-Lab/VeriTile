import bench.repairs.tritonbench_g.flash_attn.Contract
import bench.repairs.tritonbench_g.flash_attn.Proofs.SourceCorrect

namespace VeriTile.Bench.Repairs.FlashAttn.Checked
noncomputable section
set_option maxHeartbeats 2000000

/-- The checked output's sorted stride scan derives the complete injective
logical output layout; allocation observations supply shape correspondence. -/
theorem output_admitted (a : Host.Arguments) (o : Tensor) (scratch : Nat)
    (allocation : Allocations a o scratch) (order : SortedAxes) (ho : tensorChecks o order=Bool.true) :
    Layouts.Admitted (Host.shape a) o.tensor.strides := by
  obtain ⟨axes,perm,guard,_⟩:=(admitted_tensor o order ho).span
  refine ⟨axes,perm,?_⟩
  simpa only [Host.shape,allocation.outputShape,Layouts.sizes] using guard

/-- Independent complete Real theorem after the actual input/output checks.
No source-execution, source-bound or output-value fact is supplied by binding. -/
theorem whole_real (q k v o : Tensor) (qo ko vo oo : SortedAxes) (scale : ScalarABI)
    (causal : Bool) (owner : Nat→Nat→Nat→Fin 4) (scratch : Nat)
    (hi : inputChecks q k v qo ko vo scale=Bool.true) (ho : tensorChecks o oo=Bool.true)
    (allocation : Allocations (arguments q k v causal scale owner) o scratch)
    (memory : Memory ℝ) :
    WholeReal (Host.layout (arguments q k v causal scale owner) o.tensor) memory
      (scale.project scale.raw) (sourceSchedule (Host.shape (arguments q k v causal scale owner))) := by
  let a:=arguments q k v causal scale owner
  exact FlashAttn.whole_real (Host.layout a o.tensor)
    (Host.contract_shape a (admitted_inputs q k v qo ko vo scale hi causal owner))
    (output_admitted a o scratch allocation oo ho) memory (scale.project scale.raw) _ (List.Perm.refl _)

/-- Padded tensor addresses and all their bytes follow from the executed
metadata guards independently of any Real/native numerical correspondence. -/
theorem structural_safe (t : Tensor) (order : SortedAxes) (h : tensorChecks t order=Bool.true)
    (b head n d : Nat) (hb : b<t.tensor.shape.batches) (hh : head<t.tensor.shape.heads)
    (hn : n<t.tensor.shape.length) (hd : d<t.tensor.shape.dim) (j : Fin 4) :
    4*(t.tensor.storageOffset+address t.tensor.strides b head n d+1)≤t.tensor.storageBytes ∧
    t.dataPointer%4=0 ∧ t.dataPointer+4*address t.tensor.strides b head n d+j.val<2^64 := by
  exact ⟨Host.tensor_access t.tensor (admitted_tensor t order h) b head n d hb hh hn hd,
    (formed_tensor t order h b head n d hb hh (by omega) hd j).2⟩

#axiomsClean output_admitted
#axiomsClean whole_real
#axiomsClean structural_safe
end
end VeriTile.Bench.Repairs.FlashAttn.Checked
