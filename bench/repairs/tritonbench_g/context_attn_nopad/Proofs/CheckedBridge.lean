import bench.repairs.tritonbench_g.context_attn_nopad.Contract

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
set_option maxRecDepth 24000
noncomputable section

/-- Ordinary agreement of CPU int32 observations and the original metadata
memory. Payloads remain arbitrary Real values; no floating encoding is used. -/
structure MetadataObservation (c : Call) (r : Regions) (s : BlockState)
    (start length : Nat → Int) where
  (startWords lengthWords : Nat → BitVec 32)
  startCPU : ∀ b,b < c.batches → start b=(startWords b).toInt
  lengthCPU : ∀ b,b < c.batches → length b=(lengthWords b).toInt
  startMemory : ∀ b,b < c.batches → (inputs s r).start b=start b
  lengthMemory : ∀ b,b < c.batches → (inputs s r).sequence b=length b

def Labels (r : Regions) : Prop :=
  r.out≠r.query ∧ r.out≠r.key ∧ r.out≠r.value ∧ r.out≠r.start ∧ r.out≠r.sequence

theorem shape (c : Call) (r : Regions) (extent : RegionBounds) (s : BlockState)
    (start length : Nat → Int) (checked : geometryChecks c r extent)
    (metadata : metadataChecks c start length) (observed : MetadataObservation c r s start length) :
    Host.Shape c := by
  rcases checked with ⟨width,heads,k,v,o,contiguous,capacity,packed,space,se,le⟩
  have hw : c.q.dims=16 ∨ c.q.dims=32 ∨ c.q.dims=64 ∨ c.q.dims=128 := by
    simpa only [Finset.mem_insert,Finset.mem_singleton] using width
  refine ⟨hw,by rw [k],by rw [v],by rw [o],by rw [k],by rw [v],by rw [o],
    ?_,?_,?_,?_,intervals c packed,?_,?_,?_⟩
  · intro b hb; exact (capacity ⟨b,hb⟩).1
  · intro b hb; rw [k]; exact (capacity ⟨b,hb⟩).1
  · intro b hb; rw [v]; exact (capacity ⟨b,hb⟩).1
  · intro b hb; rw [o]; exact (capacity ⟨b,hb⟩).1
  · rw [o]; exact injective c.q contiguous
  · intro b hb
    have hm := (metadata.1 ⟨b,hb⟩).1
    have ho := observed.startCPU b hb
    have bound := BitVec.toInt_lt (x := observed.startWords b)
    change (observed.startWords b).toInt < (2: Int)^31 at bound
    have eq : (c.start b : Int)=(observed.startWords b).toInt := hm.symm.trans ho
    rw [← eq] at bound
    exact_mod_cast bound
  · intro b hb
    have hm := (metadata.1 ⟨b,hb⟩).2
    have ho := observed.lengthCPU b hb
    have bound := BitVec.toInt_lt (x := observed.lengthWords b)
    change (observed.lengthWords b).toInt < (2:Int)^31 at bound
    have eq : (c.length b : Int)=(observed.lengthWords b).toInt := hm.symm.trans ho
    rw [← eq] at bound
    exact_mod_cast bound

theorem memory (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (start length : Nat → Int) (checked : geometryChecks c r A.extent)
    (metadata : metadataChecks c start length) (observed : MetadataObservation c r s start length)
    (labels : Labels r) : Host.Memory c r A s := by
  have hs := shape c r A.extent s start length checked metadata observed
  rcases checked with ⟨width,heads,k,v,o,contiguous,capacity,packed,space,se,le⟩
  refine ⟨hs,?_,?_,se,le,?_,?_,?_,?_,labels.1,labels.2.1,labels.2.2.1,labels.2.2.2.1,labels.2.2.2.2⟩
  · intro b hb; exact (observed.startMemory b hb).trans (metadata.1 ⟨b,hb⟩).1
  · intro b hb; exact (observed.lengthMemory b hb).trans (metadata.1 ⟨b,hb⟩).2
  · exact bounded c.q contiguous _ (space _ (by simp))
  · rw [k]; exact bounded c.q contiguous _ (space r.key (by simp))
  · rw [v]; exact bounded c.q contiguous _ (space r.value (by simp))
  · rw [o]; exact bounded c.q contiguous _ (space r.out (by simp))

theorem complete (c : Call) (r : Regions) (extent : RegionBounds)
    (checked : geometryChecks c r extent) : c.completeGrid := by
  intro b hb
  exact (checked.2.2.2.2.2.2.1 ⟨b,hb⟩).2

/-- Allocation-window observations bind queried sizes to the regional model.
Only original storage descriptors occur; there is no expected-result field. -/
structure AllocationSizes (r : Regions) (v : Views) (A : FlatAlloc) : Prop where
  listed : A.regions=[r.query,r.key,r.value,r.out,r.start,r.sequence]
  covered : ∀ rg,rg∉A.regions → A.extent rg=0
  query : 4*A.extent r.query≤v.q.storageBytes
  key : 4*A.extent r.key≤v.k.storageBytes
  value : 4*A.extent r.value≤v.v.storageBytes
  output : 4*A.extent r.out≤v.o.storageBytes
  start : 4*A.extent r.start≤v.start.storageBytes
  length : 4*A.extent r.sequence≤v.length.storageBytes

theorem abi (c : Call) (r : Regions) (v : Views) (A : FlatAlloc) (requestedMax : Nat)
    (tensors : tensorChecks c v) (native : nativeChecks c v requestedMax)
    (sizes : AllocationSizes r v A) : NativeProgram.ABI c A := by
  refine ⟨by have := native.2.2.2.1; norm_num at *; omega,?_⟩
  intro rg
  by_cases mem : rg∈A.regions
  · rw [sizes.listed] at mem
    simp only [List.mem_cons,List.not_mem_nil,or_false] at mem
    have bound (x : View) (hx : x∈all v) : x.storageBytes < 2^63 :=
      (tensors.2.2.2.1 x hx).2.2.2.2.2.1
    rcases mem with eq|eq|eq|eq|eq|eq
    all_goals subst rg
    · exact le_trans sizes.query (le_of_lt (bound v.q (by simp [all])))
    · exact le_trans sizes.key (le_of_lt (bound v.k (by simp [all])))
    · exact le_trans sizes.value (le_of_lt (bound v.v (by simp [all])))
    · exact le_trans sizes.output (le_of_lt (bound v.o (by simp [all])))
    · exact le_trans sizes.start (le_of_lt (bound v.start (by simp [all])))
    · exact le_trans sizes.length (le_of_lt (bound v.length (by simp [all])))
  · rw [sizes.covered rg mem]; simp

#axiomsClean shape
#axiomsClean memory
#axiomsClean abi
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Checked
