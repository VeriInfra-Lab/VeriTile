import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Memory
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory
variable {α : Type}
noncomputable section

def writeTensorWhere (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) : Memory α := by
  classical
  exact scatter b (offset c) x (Finset.univ.toList.filter (fun i => decide (keep i))) s

theorem writeTensorWhere_readback (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) (i : Coord c) (hi : keep i) :
    writeTensorWhere c s b keep x b (offset c i) = x i := by
  classical
  apply scatter_readback
  · simp [hi]
  · exact offset_injective c

theorem writeTensorWhere_other (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) (r : Buffer) (o : Nat) (hr : r ≠ b) :
    writeTensorWhere c s b keep x r o = s r o := by
  classical
  exact scatter_frame _ _ _ _ _ _ _ (fun _ _ => Or.inl hr)

theorem writeTensorWhere_skip (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) (i : Coord c) (hi : ¬keep i) :
    writeTensorWhere c s b keep x b (offset c i) = s b (offset c i) := by
  classical
  apply scatter_frame
  intro j hj
  right
  intro he
  have hj' : keep j := by simpa using (List.mem_filter.mp hj).2
  exact hi (offset_injective c he ▸ hj')

theorem writeTensorWhere_outside (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) (r : Buffer) (o : Nat) (ho : c.elements ≤ o) :
    writeTensorWhere c s b keep x r o = s r o := by
  classical
  apply scatter_frame
  intro i _
  right
  have hi := offset_lt c i
  omega

#axiomsClean writeTensorWhere_readback
#axiomsClean writeTensorWhere_skip
#axiomsClean writeTensorWhere_outside
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory
