import bench.repairs.tritonbench_g.parallel_attention.Contract

namespace VeriTile.Bench.Repairs.ParallelAttention.Checked
set_option maxHeartbeats 1000000

theorem shapeCheck_iff (s : Shape) : shapeCheck s=true ↔ ShapeContract s := by
  simp only [shapeCheck,decide_eq_true_eq]
  exact ⟨fun ⟨a,b,c,d,e⟩=>⟨a,b,c,d,e⟩,fun h=>⟨h.key_positive,h.key_limit,h.value_positive,h.key_tile,h.value_tile⟩⟩

theorem machineCheck_iff (s : Shape) : machineCheck s=true ↔ Machine.Admission s := by
  simp only [machineCheck,decide_eq_true_eq,Machine.Admission,Machine.Empty]
  constructor
  · rintro (h|h|h|⟨a,b,c⟩)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr (Or.inl h))
    · exact Or.inl (Or.inr (Or.inr h))
    · exact Or.inr ⟨a,b,c⟩
  · rintro ((h|h|h)|h)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
    · exact Or.inr (Or.inr (Or.inr ⟨h.heads_axis,h.time_axis,h.formed⟩))

theorem checkCall_iff (c : Call) : checkCall c=.ok () ↔ Domain c := by
  have domain : Domain c ↔ ShapeContract c.shape ∧ TensorKind c.q ∧ TensorKind c.k ∧ TensorKind c.v ∧
      MetadataBinding c ∧ (DenseStrides c.shape c.shape.K c.q ∧ DenseStrides c.shape c.shape.K c.k ∧
      DenseStrides c.shape c.shape.V c.v) ∧ (c.q.tensor.device=c.k.tensor.device ∧ c.q.tensor.device=c.v.tensor.device) ∧
      Stored c.q (c.shape.B*c.shape.H*c.shape.T*c.shape.K) ∧
      Stored c.k (c.shape.B*c.shape.H*c.shape.T*c.shape.K) ∧
      Stored c.v (c.shape.B*c.shape.H*c.shape.T*c.shape.V) ∧ Machine.Admission c.shape ∧ PointerRoom c := by
    constructor
    · intro h;exact ⟨h.shape,h.qKind,h.kKind,h.vKind,h.metadata,h.strides,h.devices,h.qStorage,h.kStorage,h.vStorage,h.admission,h.room⟩
    · rintro ⟨a,b,d,e,f,g,h,i,j,k,l,n⟩;exact ⟨a,b,d,e,f,g,h,i,j,k,l,n⟩
  rw [domain]
  simp only [checkCall]
  split_ifs <;> simp_all [shapeCheck_iff,machineCheck_iff] <;> aesop

theorem check_iff (c : ClassifiedCall) : check c=.ok () ↔ Admitted c := by
  cases c <;> simp [check,Admitted,checkCall_iff]

theorem rejected_iff (c : ClassifiedCall) : (∃ e,check c=.error e) ↔ ¬Admitted c := by
  rw [←check_iff]
  cases check c <;> simp

theorem inputs_valid (c : Call) : (inputs c).Valid c.shape := ⟨rfl,rfl,rfl⟩

/-- No current-device equality is required: all three launch/host paths enter
`torch.cuda.device(q.device)` and restore the caller's previous device. -/
def selectedDevice (c : Call) := c.q.tensor.device.index

theorem stored_address (t : TensorView) (words i : Nat) (h : Stored t words) (hi : i<words) :
    4*(t.pointer/4+i)=t.pointer+4*i ∧
    t.storageBase≤t.pointer+4*i ∧ t.pointer+4*i+4≤t.storageBase+t.capacity := by
  obtain ⟨aligned,lo,hi',limit⟩:=h (by omega)
  constructor
  · omega
  · omega

theorem pointer_room (c : Call) (d : Domain c) (active : Machine.Active c.shape)
    (t : TensorView) (ht : t=c.q ∨ t=c.k ∨ t=c.v)
    (offset : Nat) (ho : offset<Machine.formedLimit c.shape) : t.pointer+4*offset<2^63 := by
  have room:=d.room
  unfold PointerRoom Machine.Empty at room
  rcases active with ⟨hb,hh,hT⟩
  rcases room with impossible|⟨hq,hk,hv⟩
  · omega
  · rcases ht with rfl|rfl|rfl <;> omega

#axiomsClean check_iff
#axiomsClean rejected_iff
#axiomsClean stored_address
#axiomsClean pointer_room
#specNonCircular Domain avoiding [combinedRun,HostAPI.forward,HostAPI.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ParallelAttention.Checked
