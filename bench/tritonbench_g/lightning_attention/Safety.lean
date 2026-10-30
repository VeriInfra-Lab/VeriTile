import bench.tritonbench_g.lightning_attention.Tensor
import bench.tritonbench_g.lightning_attention.Proofs.AddressBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton

def AddressSafety {phase} (r : Request phase) : Prop :=
  ∀ x∈reads r ++ (List.finRange (outputs phase)).map r.result,
  ∀ width head row feature, width≤max r.dimensions.D r.dimensions.E →
    head<r.dimensions.B*r.dimensions.H → row<r.dimensions.N+128 → feature<width →
    Host.formedOffset r.dimensions width head row feature<2^31 ∧
    x.tensor.dataPtr.toNat+4*Host.formedOffset r.dimensions width head row feature<2^63 ∧
    (x.tensor.dataPtr.toNat+4*Host.formedOffset r.dimensions width head row feature)%4=0

theorem addresses {phase} (r : Request phase) (h : Contract r) : AddressSafety r := by
  intro x hx width head row feature hw hh hr hf
  have bound := Host.formed_offset r.dimensions width head row feature hw hh hr hf
  have checked := h.pointer x hx
  refine ⟨lt_trans bound h.dimensions.machine,by omega,?_⟩
  simp [Nat.add_mod,checked.1]

/-- These source-row lemmas cover all formed pointers, including false mask
lanes, the final forward advances, and the reverse initial pointer vector. -/
theorem source_rows (d : Host.Dimensions) :
    (∀ b lane,b≤d.NT → lane<64 → b*64+lane<d.N+128) ∧
    (∀ b sub lane,b<d.NT → sub<2 → lane<32 → (b*2+sub)*32+lane<d.N+128) ∧
    (∀ step lane,step≤d.NT*2 → lane<32 → step*32+lane<d.N+128) ∧
    (∀ lane,lane<32 → d.NT*64+lane<d.N+128) ∧
    (∀ b sub lane,b<d.NT → sub<2 → lane<32 → d.NT*64-(b*2+sub+1)*32+lane<d.N+128) :=
  ⟨Host.formed_row d,Host.forward_sub_row d,Host.forward_advance_row d,
    Host.reverse_initial_row d,Host.reverse_sub_row d⟩

#axiomsClean addresses
#axiomsClean source_rows
#axiomsClean Host.formed_intermediates
#axiomsClean Host.dimension_machine
#axiomsClean Host.mask_index_machine
#axiomsClean Host.reverse_signed_row
#axiomsClean Host.reverse_pointer_step
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
