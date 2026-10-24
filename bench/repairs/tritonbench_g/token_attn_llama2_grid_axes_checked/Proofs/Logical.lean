import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Frame
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Addresses

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000

def queryIndex (t : TensorCall) (b h d : Nat) := (b*t.call.heads+h)*t.call.dim+d
def keyIndex (t : TensorCall) (words : Nat → Word) (b h n d : Nat) :=
  ((metadata t words).page b n*t.call.kvHeads+h/(t.call.heads/t.call.kvHeads))*t.call.dim+d

theorem query_index (t : TensorCall) (header : Header t) (strides : DenseChecks t)
    (b h d : Nat) : b*t.call.qBatch+h*t.call.qHead+d*t.call.qDim=queryIndex t b h d := by
  have layout := dense t header strides
  simp [queryIndex,layout.q.1,layout.q.2.1,layout.q.2.2,Nat.add_mul,Nat.mul_assoc]

theorem key_index (t : TensorCall) (words : Nat → Word) (header : Header t) (strides : DenseChecks t)
    (b h n d : Nat) :
    (metadata t words).page b n*t.call.kBatch+(h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d*t.call.kDim=
      keyIndex t words b h n d := by
  have layout := dense t header strides
  simp [keyIndex,layout.k.1,layout.k.2.1,layout.k.2.2,Nat.add_mul,Nat.mul_assoc]

theorem query_tensor {A} [Inhabited A] (t : TensorCall) (words : Nat → Word)
    (before : Cells A) (contract : InputContract t words) (launch : active t)
    (b h : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads) (d : Fin t.call.dim) :
    query t.call (buffers t before) b h d=logical before t.q (queryIndex t b h d.val) := by
  have domain := valid t words contract launch
  have bound := domain.queryBounds b hb h hh d.val d.isLt
  have equal := query_index t contract.1 (contract.2.2.2 launch).1 b h d.val
  have contiguous : t.q.tensor.metadata.isContiguous=Bool.true :=
    contract.1.contiguous t.q.observation (by simp [TensorCall.observations])
  simp only [query,finiteRead,dif_pos bound,buffers]
  rw [equal]
  exact payload_read before t.q contiguous _ (by simpa only [←equal] using bound)

theorem key_tensor {A} [Inhabited A] (t : TensorCall) (words : Nat → Word)
    (before : Cells A) (contract : InputContract t words) (launch : active t)
    (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads)
    (hn : n<(metadata t words).len b) (d : Fin t.call.dim) :
    key t.call (metadata t words) (buffers t before) b h n d=
      logical before t.k (keyIndex t words b h n d.val) := by
  have domain := valid t words contract launch
  have page := (domain.pages ⟨b,hb⟩ n hn).2
  have head := grouped_head t.call (metadata t words) domain h hh
  have bound := domain.keyBounds _ page _ head d.val d.isLt
  have equal := key_index t words contract.1 (contract.2.2.2 launch).1 b h n d.val
  have contiguous : t.k.tensor.metadata.isContiguous=Bool.true :=
    contract.1.contiguous t.k.observation (by simp [TensorCall.observations])
  simp only [key,finiteRead,dif_pos bound,buffers]
  rw [equal]
  exact payload_read before t.k contiguous _ (by simpa only [←equal] using bound)

theorem logical_word (before : Cells Word) (t : Payload)
    (ordinary : t.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical before t i=(VeriTile.Torch.FP32.logicalValue
      (fun address => ⟨(before address).read .fp32⟩) t.tensor t.laws i).bits := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,ordinary]

#axiomsClean query_index
#axiomsClean key_index
#axiomsClean query_tensor
#axiomsClean key_tensor
#axiomsClean logical_word
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
