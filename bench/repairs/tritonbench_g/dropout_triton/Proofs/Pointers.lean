import bench.repairs.tritonbench_g.dropout_triton.Proofs.Memory
import bench.tritonbench_g.vector_addition.Proofs.FiniteIndices

namespace VeriTile.Bench.Repairs.DropoutTriton.Physical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition DTypeScope TypedHost Finite
noncomputable section

/-- Input Tensor API binding for a nonempty contiguous view. Storage extent,
actual data_ptr, and offset are ordinary Tensor facts; no arithmetic result is
postulated. Empty calls need no window or pointer fact. -/
structure TensorWindow (metadata : DropoutTriton.TensorMetadata) (dtype : DType) where
  view : View
  type : view.dtype = dtype
  device : view.device = metadata.device.index
  pointer : 0 < metadata.numel → view.address 0 = metadata.dataPtr.toNat
  bounds : 0 < metadata.numel → view.Valid metadata.numel metadata.numel

/-- The source's actual signed comparison chooses the stored lane jobs. -/
def nativeJobs (n pid : Nat) : List Nat :=
  ((List.range 1024).filter (fun lane => DropoutTriton.mask32 n pid lane)).map
    (fun lane => (DropoutTriton.offset32 pid lane).toInt.toNat)

def nativeBatches (n : Nat) : List (List Nat) :=
  (List.range (gridCount n)).map (nativeJobs n)

theorem native_batches_exact (x keep : DropoutTriton.TensorMetadata) (h : TensorContract x keep) :
    nativeBatches x.numel = sourceBatches x.numel := by
  unfold nativeBatches sourceBatches
  apply List.map_congr_left
  intro pid hp
  have hp' : pid < gridCount x.numel := List.mem_range.mp hp
  unfold nativeJobs programLanes
  rw [List.filter_map]
  have hm : (List.range 1024).filter (fun lane => DropoutTriton.mask32 x.numel pid lane) =
      (List.range 1024).filter (fun lane => decide (pid*1024+lane < x.numel)) := by
    apply List.filter_congr
    intro lane hl
    exact (DropoutTriton.native_indices x keep h pid lane hp' (List.mem_range.mp hl)).2
  rw [hm]
  apply List.map_congr_left
  intro lane hl
  have bound := List.mem_range.mp (List.mem_filter.mp hl).1
  rw [(DropoutTriton.native_indices x keep h pid lane hp' bound).1]
  rfl

/-- Only active pointers need to address storage. Rounded inactive tail lanes
are never granted an out-of-storage pointer premise. -/
theorem active_pointer (v : View) (n pid lane : Nat) (hn : n < 2^31)
    (valid : v.Valid n n) (hp : pid < gridCount n) (hl : lane < 1024)
    (active : pid*1024+lane < n) :
    (Finite.pointer64 v pid lane).toNat = v.address (pid*1024+lane) := by
  have access := valid_access v n n (pid*1024+lane) valid active
  have hb : v.address (pid*1024+lane) < 2^64 := by
    have positive := bytes_positive v.dtype
    unfold View.finish at access
    omega
  unfold Finite.pointer64
  rw [source_index_sign_extend n pid lane hn hp hl, BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat, ←address_linear, Nat.mod_eq_of_lt hb]

/-- Actual mask, pointer, alignment and byte bounds for each source access. -/
theorem admitted_input_address (x keep : DropoutTriton.TensorMetadata) (h : TensorContract x keep)
    (v : View) (dtype : DType) (width : bytes dtype = 4)
    (type : v.dtype = dtype) (valid : v.Valid x.numel x.numel)
    (aligned : v.address 0 % 4 = 0) (pid lane : Nat)
    (hp : pid < gridCount x.numel) (hl : lane < 1024)
    (active : DropoutTriton.mask32 x.numel pid lane = Bool.true) :
    (Finite.pointer64 v pid lane).toNat = v.address (pid*1024+lane) ∧
    v.address (pid*1024+lane) % 4 = 0 ∧
    v.storageBase ≤ v.address (pid*1024+lane) ∧
    v.finish (pid*1024+lane) ≤ v.storageBase+v.storageBytes ∧
    v.finish (pid*1024+lane) ≤ 2^64 := by
  rw [(DropoutTriton.native_indices x keep h pid lane hp hl).2] at active
  have ha := of_decide_eq_true active
  have va : v.Aligned := by simpa [View.Aligned, type, width] using aligned
  exact ⟨active_pointer v x.numel pid lane h.extent valid hp hl ha,
    by simpa [type, width] using address_aligned v va (pid*1024+lane),
    valid_access v x.numel x.numel (pid*1024+lane) valid ha⟩

/-- The finite source loop loads each whole active tile before storing it.
The typed execution proof separately retains all inactive register arithmetic. -/
def runNative (ops : Memory M A K) (compute : A → K → A) (x keep out : View)
    (n : Nat) (before : M) : M :=
  runBatches ops compute x keep out (nativeBatches n) before

theorem complete_native (ops : Memory M A K) (compute : A → K → A)
    (x keep : DropoutTriton.TensorMetadata) (h : TensorContract x keep)
    (vx vk vo : View) (before : M)
    (xt : vx.dtype = .fp32) (ot : vo.dtype = .fp32)
    (xb : vx.Valid x.numel x.numel) (kb : vk.Valid x.numel x.numel) (ob : vo.Valid x.numel x.numel)
    (freshX : StorageDisjoint vo vx) (freshKeep : StorageDisjoint vo vk) :
    (∀ i, i < x.numel → readX ops (runNative ops compute vx vk vo x.numel before) vo i =
      compute (readX ops before vx i) (readK ops before vk i)) ∧
    (∀ a, ¬vo.active x.numel a →
      ops.byte (runNative ops compute vx vk vo x.numel before) a = ops.byte before a) := by
  unfold runNative
  rw [native_batches_exact x keep h]
  exact source_complete ops compute vx vk vo x.numel before xt ot xb kb ob freshX freshKeep

#axiomsClean native_batches_exact
#axiomsClean active_pointer
#axiomsClean admitted_input_address
#axiomsClean complete_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Physical
