import bench.repairs.tritonbench_g.kv_cache_filling.IO
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantExecution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling
open VeriTile.Triton
noncomputable section

namespace SourceTensor

theorem source_read {α : Type} [Inhabited α] (decode : (Fin 4 → Byte) → α)
    (t : TensorInputs) (r : Request) (m : Nat → Byte) (schedule : Schedule t r m)
    (scratch : FP.Structural.State α) (j : Job) (hj : j ∈ jobs r) :
    ((initial decode t r m schedule scratch j).mem "src"
      ((initial decode t r m schedule scratch j).pids 0)).read .int = (sourceRow r j : Int) := by
  simpa [initial,WholeBacking.memory] using schedule.sourceRead j hj

theorem destination_read {α : Type} [Inhabited α] (decode : (Fin 4 → Byte) → α)
    (t : TensorInputs) (r : Request) (m : Nat → Byte) (schedule : Schedule t r m)
    (scratch : FP.Structural.State α) (j : Job) (hj : j ∈ jobs r) :
    ((initial decode t r m schedule scratch j).mem "dst"
      ((initial decode t r m schedule scratch j).pids 0)).read .int = (destinationRow r j : Int) := by
  simpa [initial,WholeBacking.memory] using schedule.destinationRead j hj

#axiomsClean source_read
#axiomsClean destination_read
end SourceTensor

namespace WholeBacking

theorem input_row {α : Type} [Inhabited α] (decode : (Fin 4 → Byte) → α)
    (t : TensorInputs) (r : Request) (m : Nat → Byte) (schedule : SourceTensor.Schedule t r m)
    (scratch : FP.Structural.State α) (j : Job) (key : Bool) :
    QuantExecution.inputRow (Shared.parameters r)
      (SourceTensor.initial decode t r m schedule scratch j) (sourceRow r j) key =
        SourceTensor.row decode t r m j key := by
  funext i
  cases key <;> simp [QuantExecution.inputRow,SourceTensor.initial,memory,
    CopyExecution.input,CopyExecution.width,Shared.parameters,SourceTensor.row]

#axiomsClean input_row
end WholeBacking

namespace SourceBacking

theorem mem_stores {α : Type} [Inhabited α] (r : Request)
    (states : Job → FP.Structural.State α) (s : Store (Cell α Byte)) :
    s ∈ stores r states ↔ ∃ j ∈ jobs r,
      s.address ∈ rowAddresses r j ∧ s.value = project r (states j) s.address := by
  simp only [stores,List.mem_flatMap,List.mem_map]
  constructor
  · rintro ⟨j,hj,a,ha,e⟩
    subst s
    exact ⟨j,hj,ha,rfl⟩
  · rintro ⟨j,hj,ha,hv⟩
    exact ⟨j,hj,s.address,ha,by cases s; simp_all⟩

theorem readback {α : Type} [Inhabited α] (r : Request) (d : Domain r) (l : Layout r)
    (states : Job → FP.Structural.State α) (encode : α → Nat → Byte) (m : Nat → Byte)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) :
    memory r l states encode m (bytePtr r l a b) = cellByte encode (project r (states j) a) b := by
  apply runStores_readback
  · let s : Store (Cell α Byte) := ⟨a,project r (states j) a⟩
    refine ⟨⟨physical (bytePtr r l a b),cellByte encode s.value b⟩,?_,rfl⟩
    exact List.mem_flatMap.mpr ⟨s,(mem_stores ..).mpr ⟨j,hj,ha,rfl⟩,
      (mem_expandStore ..).mpr ⟨b,hb,rfl,rfl⟩⟩
  · intro t ht he
    obtain ⟨s,hs,ht'⟩ := List.mem_flatMap.mp ht
    obtain ⟨b',hb',hta,htv⟩ := (mem_expandStore ..).mp ht'
    obtain ⟨j',hj',ha',hsv⟩ := (mem_stores ..).mp hs
    have va : ValidByte r a b := ⟨(row_address_bound r d j hj a ha).1,
      (mem_rowAddresses r j a).mp ha |>.2.1,(mem_rowAddresses r j a).mp ha |>.2.2,hb⟩
    have va' : ValidByte r s.address b' := ⟨(row_address_bound r d j' hj' s.address ha').1,
      (mem_rowAddresses r j' s.address).mp ha' |>.2.1,(mem_rowAddresses r j' s.address).mp ha' |>.2.2,hb'⟩
    have hp : bytePtr r l s.address b' = bytePtr r l a b :=
      congrArg Address.row (hta.symm.trans he)
    obtain ⟨hsa,hbb⟩ := bytePtr_injective r l s.address a b' b va' va hp
    have hsame : j'=j := destinationRow_injective r d j' j hj' hj (by
      rw [← ((mem_rowAddresses r j' s.address).mp ha').1,
        ← ((mem_rowAddresses r j a).mp ha).1,hsa])
    simpa [hsa,hbb,hsame,hsv] using htv

theorem tensor_readback {α : Type} [Inhabited α] (t : TensorInputs) (r : Request)
    (h : InputContract t r) (l : Layout r) (ins : InputLayout r l)
    (binding : Tensor.Bindings t r l ins) (states : Job → FP.Structural.State α)
    (encode : α → Nat → Byte) (m : Nat → Byte)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) :
    Tensor.byte (Tensor.output t a.role) (memory r l states encode m)
      (a.row*r.width a.role+a.column) b = cellByte encode (project r (states j) a) b := by
  have valid : ValidByte r a b := ⟨(row_address_bound r (Native.domain h) j hj a ha).1,
    (mem_rowAddresses r j a).mp ha |>.2.1,(mem_rowAddresses r j a).mp ha |>.2.2,hb⟩
  rw [← Tensor.output_read t r h l ins binding _ a b valid]
  exact readback r (Native.domain h) l states encode m j hj a ha b hb

theorem frame {α : Type} [Inhabited α] (r : Request) (l : Layout r)
    (states : Job → FP.Structural.State α) (encode : α → Nat → Byte) (m : Nat → Byte)
    (p : Nat) (outside : ∀ j ∈ jobs r, ∀ a ∈ rowAddresses r j, ∀ b,
      b < elementBytes r a.role → bytePtr r l a b ≠ p) :
    memory r l states encode m p = m p := by
  apply runStores_frame
  intro t ht he
  obtain ⟨s,hs,ht'⟩ := List.mem_flatMap.mp ht
  obtain ⟨b,hb,ha,_⟩ := (mem_expandStore ..).mp ht'
  obtain ⟨j,hj,hsa,_⟩ := (mem_stores ..).mp hs
  exact outside j hj s.address hsa b hb (congrArg Address.row (ha.symm.trans he))

#axiomsClean readback
#axiomsClean tensor_readback
#axiomsClean frame
end SourceBacking
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling
