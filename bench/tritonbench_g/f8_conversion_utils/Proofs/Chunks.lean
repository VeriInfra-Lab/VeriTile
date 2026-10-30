import bench.tritonbench_g.f8_conversion_utils.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_NativeArithmetic
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteSchedule
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteIndices

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Chunking
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost Finite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

def chunkBatches (n k : Nat) : List (List Nat) :=
  (Finite.sourceBatches (chunkLength n k)).map fun lanes => lanes.map (chunkStart k+·)
def batches (n : Nat) : List (List Nat) :=
  (List.range (chunkCount n)).flatMap (chunkBatches n)

theorem chunk_small (n k : Nat) : chunkLength n k < 2^31 := by
  have h := Nat.min_le_left chunkSize (n - chunkStart k)
  norm_num [chunkLength,chunkSize] at h ⊢

theorem chunk_grid_supported (n k : Nat) : Native.gridSupported (chunkLength n k) := by
  have h := chunk_small n k
  unfold Native.gridSupported Native.grid
  norm_num at h ⊢
  omega

theorem shifted_membership (lists : List (List Nat)) (start i : Nat) :
    i ∈ (lists.map (fun lanes => lanes.map (start + ·))).flatten ↔
      ∃ j, j ∈ lists.flatten ∧ start+j=i := by
  constructor
  · intro hi
    obtain ⟨mapped,hm,hi⟩ := List.mem_flatten.mp hi
    obtain ⟨lanes,hl,rfl⟩ := List.mem_map.mp hm
    obtain ⟨j,hj,he⟩ := List.mem_map.mp hi
    exact ⟨j,List.mem_flatten.mpr ⟨lanes,hl,hj⟩,he⟩
  · rintro ⟨j,hj,he⟩
    obtain ⟨lanes,hl,hj⟩ := List.mem_flatten.mp hj
    exact List.mem_flatten.mpr ⟨_,List.mem_map.mpr ⟨lanes,hl,rfl⟩,
      List.mem_map.mpr ⟨j,hj,he⟩⟩

theorem chunk_membership (n k i : Nat) :
    i ∈ (chunkBatches n k).flatten ↔
      ∃ j, j < chunkLength n k ∧ chunkStart k+j=i := by
  rw [chunkBatches,shifted_membership,Finite.source_batches_flatten]
  simp only [Finite.mem_activeLanes]

theorem batches_membership (n i : Nat) :
    i ∈ (batches n).flatten ↔
      ∃ k, k < chunkCount n ∧ ∃ j, j < chunkLength n k ∧ chunkStart k+j=i := by
  constructor
  · intro hi
    obtain ⟨lanes,hl,hi⟩ := List.mem_flatten.mp hi
    obtain ⟨k,hk,hl⟩ := List.mem_flatMap.mp hl
    exact ⟨k,List.mem_range.mp hk,(chunk_membership n k i).mp (List.mem_flatten.mpr ⟨lanes,hl,hi⟩)⟩
  · rintro ⟨k,hk,hj⟩
    obtain ⟨lanes,hl,hi⟩ := List.mem_flatten.mp ((chunk_membership n k i).mpr hj)
    exact List.mem_flatten.mpr ⟨lanes,List.mem_flatMap.mpr ⟨k,List.mem_range.mpr hk,hl⟩,hi⟩

theorem full_coverage (n i : Nat) : i ∈ (batches n).flatten ↔ i < n := by
  rw [batches_membership]
  constructor
  · rintro ⟨k,hk,j,hj,rfl⟩
    dsimp [chunkLength,chunkStart,chunkSize] at hj ⊢
    omega
  · intro hi
    refine ⟨i/chunkSize,?_,i%chunkSize,?_,?_⟩
    · unfold chunkCount chunkSize
      omega
    · unfold chunkLength chunkStart chunkSize
      omega
    · simpa [chunkStart,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i chunkSize

theorem chunk_active (n k j : Nat) (hj : j < chunkLength n k) : chunkStart k+j<n := by
  dsimp [chunkLength,chunkStart,chunkSize] at hj ⊢
  omega

/-- The unchanged source int32 arithmetic is exact for every launched lane,
including masked tails, because the host bounds each actual scalar N. -/
theorem finite_indices (n k pid lane : Nat)
    (hp : pid < Native.grid (chunkLength n k)) (hl : lane < 1024) :
    (Native.offset pid lane).toInt=(pid*1024+lane : Nat) ∧
    Native.mask (chunkLength n k) pid lane=decide (pid*1024+lane < chunkLength n k) :=
  Native.bounded_arithmetic _ _ _ (chunk_small n k) hp hl

def sliceView (v : View) (start : Nat) : View :=
  {v with storageOffset := v.storageOffset+start}

theorem slice_address (v : View) (start i : Nat) :
    (sliceView v start).address i=v.address (start+i) := by
  simp [sliceView,View.address,Nat.add_assoc]

/-- Generic slice address arithmetic keeps concrete host chunk constants out
of the kernel's definitional-equality checks. -/
theorem slice_pointer_sum (v : View) (start i : Nat) :
    (sliceView v start).address 0+i*bytes v.dtype=v.address (start+i) := by
  simp [sliceView,View.address,Nat.add_mul,Nat.add_assoc]

/-- Generic finite pointer lemma, parametrized by the actual launch count and
already-bounded active address. It does not require padded allocation space. -/
theorem pointer_bounded (count pid lane data width : Nat)
    (hn : count<2^31) (hp : pid<Native.grid count) (hl : lane<1024)
    (fits : data+(pid*1024+lane)*width<2^64) :
    Native.pointer data width pid lane=data+(pid*1024+lane)*width := by
  change (BitVec.ofNat 64 data + (offset32 pid lane).signExtend 64 * BitVec.ofNat 64 width).toNat = _
  rw [source_index_sign_extend count pid lane hn hp hl,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]

/-- Only active pointer values need an exact natural address. A masked tail
requires no padded allocation and introduces no artificial native-safe bound. -/
theorem active_pointer (v : View) (n k pid lane : Nat) (valid : v.Valid n n)
    (hp : pid < Native.grid (chunkLength n k)) (hl : lane < 1024)
    (active : pid*1024+lane < chunkLength n k) :
    Native.pointer ((sliceView v (chunkStart k)).address 0) (bytes v.dtype) pid lane =
      v.address (chunkStart k+(pid*1024+lane)) := by
  have hi := chunk_active n k (pid*1024+lane) active
  have ha := valid_access v n n (chunkStart k+(pid*1024+lane)) valid hi
  have hb : v.address (chunkStart k+(pid*1024+lane))<2^64 := by
    have hpos := bytes_positive v.dtype
    unfold View.finish at ha
    omega
  have he := slice_pointer_sum v (chunkStart k) (pid*1024+lane)
  have fits : (sliceView v (chunkStart k)).address 0+(pid*1024+lane)*bytes v.dtype<2^64 := by
    rw [he]
    exact hb
  exact (pointer_bounded (chunkLength n k) pid lane _ _ (chunk_small n k) hp hl fits).trans he

#axiomsClean chunk_small
#axiomsClean chunk_grid_supported
#axiomsClean full_coverage
#axiomsClean finite_indices
#axiomsClean pointer_bounded
#axiomsClean slice_pointer_sum
#axiomsClean active_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Chunking

namespace VeriTile.Bench.Repairs.F8ConversionUtils.NativeIndex
abbrev grid := VeriTile.Bench.TritonBenchG.VectorAddition.Native.grid
abbrev mask := VeriTile.Bench.TritonBenchG.VectorAddition.Native.mask
abbrev pointer := VeriTile.Bench.TritonBenchG.VectorAddition.Native.pointer
abbrev offset := VeriTile.Bench.TritonBenchG.VectorAddition.Native.offset
end VeriTile.Bench.Repairs.F8ConversionUtils.NativeIndex

namespace VeriTile.Bench.Repairs.F8ConversionUtils
abbrev chunkBatches := Chunking.chunkBatches
abbrev batches := Chunking.batches
end VeriTile.Bench.Repairs.F8ConversionUtils
