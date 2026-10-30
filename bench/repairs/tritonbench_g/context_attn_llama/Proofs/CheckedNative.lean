import bench.repairs.tritonbench_g.context_attn_llama.Proofs.StorageBridge
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeProgramBridge

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def widths : Native.Widths := ⟨.int32,.int32,.int32,.int32,.int32⟩
def nativeBytes (B : Bytes.Windows A.extent) : Native.Bytes A :=
  ⟨B.base,fun _ => 4,fun _ => by norm_num,B.fits⟩

theorem fits32 (n : Nat) (w : BitVec 32) (h : (n : Int)=w.toInt) : n < 2^31 := by
  have bound := BitVec.toInt_lt (x := w)
  change w.toInt < (2:Int)^31 at bound
  rw [←h] at bound
  exact_mod_cast bound

theorem abi (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc) (B : Bytes.Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (guard : InputContract c d v A.extent requestedMax start sequence prompt request pages)
    (before : Bytes.Memory) (observed : Bytes.Observed c B before start sequence prompt request pages)
    (storage : StorageBinding c v B) (needed : Bytes.Needed c) : Native.ABI c A widths (nativeBytes B) := by
  obtain ⟨tensors,geometry,native,metadata⟩ := guard
  have qBudget := native.2.2.2.2.1 v.q (by simp)
  have kBudget := native.2.2.2.2.2.1 v.k (by simp)
  rcases geometry with ⟨strides,reqRow,reqToken,kvPos,groupPos,grouped,capacity,packed,qSpace,kSpace,metaSpace,pageSpace⟩
  have dimPos : 0 < c.layout.D := by have h := c.channels; simp only [List.mem_cons,List.not_mem_nil,or_false] at h; omega
  have headLe : c.layout.heads≤c.layout.heads*c.layout.D := Nat.le_mul_of_pos_right _ dimPos
  have groupLe : c.layout.kvGroup≤c.layout.heads := by rw [grouped]; exact Nat.le_mul_of_pos_left _ kvPos
  have heads : c.layout.heads < 2^63 := by have := qBudget.2.1; omega
  have group : c.layout.kvGroup < 2^63 := lt_of_le_of_lt groupLe heads
  refine ⟨?_,by have := native.2.2.2.1; omega,heads,group,?_,?_,?_,?_,?_,?_,
    by intros; rfl,rfl,rfl,rfl,rfl,rfl,?_⟩
  · have maxBound := native.1
    have maxLe := native.2.1
    rcases c.blockM with bm|bm <;> simp only [Call.blocks,bm] <;> omega
  · intro n member
    rcases strides with ⟨qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd⟩
    simp only [qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd,reqRow,reqToken,
      List.mem_cons,List.not_mem_nil,or_false] at member
    have qrow := qBudget.2.1
    have dim := qBudget.2.2
    have krow := kBudget.2.1
    have table := native.2.2.2.2.2.2.2.2
    rcases member with h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> omega
  · intro b hb
    exact fits32 _ _ ((metadata.1 ⟨b,hb⟩).1.symm.trans (observed.start b hb).symm)
  · intro b hb
    exact fits32 _ _ ((metadata.1 ⟨b,hb⟩).2.1.symm.trans (observed.sequence b hb).symm)
  · intro b hb
    exact fits32 _ _ ((metadata.1 ⟨b,hb⟩).2.2.1.symm.trans (observed.prompt b hb).symm)
  · intro b hb
    exact fits32 _ _ ((metadata.1 ⟨b,hb⟩).2.2.2.1.symm.trans (observed.request b hb).symm)
  · intro b hb j hj
    exact fits32 _ _ (((metadata.1 ⟨b,hb⟩).2.2.2.2 ⟨j,hj⟩).symm.trans (observed.pages _).symm)
  · exact physical_separate c d v B tensors
      ⟨strides,reqRow,reqToken,kvPos,groupPos,grouped,capacity,packed,qSpace,kSpace,metaSpace,pageSpace⟩ storage needed

theorem full_metadata (c : Call) (B : Bytes.Windows extent) (before : Bytes.Memory)
    (start sequence prompt request pages : Nat → Int) (h : Bytes.Observed c B before start sequence prompt request pages)
    (guard : metadataChecks c start sequence prompt request pages) : Native.FullMetadata c (Bytes.initial c B before) := by
  refine ⟨Bytes.metadata c B before start sequence prompt request pages h guard,?_,?_⟩
  · intro b hb
    simpa [Bytes.initial,Bytes.metadataRegions] using (h.request b hb).trans (guard.1 ⟨b,hb⟩).2.2.2.1
  · intro b hb j hj
    simpa [Bytes.initial,Bytes.metadataRegions] using (h.pages _).trans ((guard.1 ⟨b,hb⟩).2.2.2.2 ⟨j,hj⟩)

theorem source_native (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc) (B : Bytes.Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (before : Bytes.Memory) (observed : Bytes.Observed c B before start sequence prompt request pages)
    (storage : StorageBinding c v B) (labels : Labels c) (needed : Bytes.Needed c) :
    Native.GridBridge c A (Bytes.layoutState B before) widths (nativeBytes B)
      (Bytes.backend c) (Bytes.initial c B before) := by
  have guard := (checkInputs_iff c d v A.extent requestedMax start sequence prompt request pages).mp checked
  exact Native.grid_bridge c A (Bytes.layoutState B before)
    (valid c d A _ start sequence prompt request pages guard.2.1 guard.2.2.2
      (Bytes.observation c B before start sequence prompt request pages observed) labels)
    widths (nativeBytes B) (abi c d v A B requestedMax start sequence prompt request pages guard before observed storage needed)
    (Bytes.backend c) (Bytes.support c) (Bytes.initial c B before)
    (full_metadata c B before start sequence prompt request pages observed guard.2.2.2)

#axiomsClean abi
#axiomsClean source_native
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Checked
