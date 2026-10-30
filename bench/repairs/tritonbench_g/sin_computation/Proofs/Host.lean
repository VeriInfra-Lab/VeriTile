import bench.repairs.tritonbench_g.sin_computation.Proofs.Execution

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Binding
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Native
export VeriTile.Torch.FP32 (Tensor TensorLaws)
abbrev Word := FP.Value .fp32
noncomputable section
set_option maxHeartbeats 6000000

/-- A logical prefix retains the actual backing storage and storage offset. -/
def window (t : Tensor) (laws : TensorLaws t)
    (c : t.metadata.isContiguous = Bool.true) (n : Nat) (cap : n ≤ t.metadata.numel) : DataView where
  storage := t.storage
  offset := t.storageOffset
  elements := n
  bounds := by
    intro positive
    have bound := VeriTile.Torch.FP32.contiguous_capacity t laws c (by omega)
    omega

theorem window_address (t : Tensor) (laws : TensorLaws t)
    (c : t.metadata.isContiguous = Bool.true) (n : Nat) (cap : n ≤ t.metadata.numel)
    (i : Nat) (hi : i<n) :
    address (window t laws c n cap) i =
      (VeriTile.Torch.FP32.dataView t laws c).address i := by
  have hn : n ≠ 0 := by omega
  have ht : t.metadata.numel ≠ 0 := by omega
  simp [address,window,VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,hn,ht]

theorem actual_pointer (t : Tensor) (laws : TensorLaws t)
    (c : t.metadata.isContiguous = Bool.true) (n : Nat) (cap : n ≤ t.metadata.numel)
    (positive : 0<n) : (view (window t laws c n cap) 0).pointer = t.dataPtr := by
  unfold view
  rw [window_address t laws c n cap 0 positive]
  exact VeriTile.Torch.FP32.actual_pointer t laws c (by omega)

theorem base_pointer (t : Tensor) (laws : TensorLaws t)
    (c : t.metadata.isContiguous = Bool.true) (n : Nat) (cap : n ≤ t.metadata.numel)
    (positive : 0<n) : (window t laws c n cap).base = t.dataPtr.toNat := by
  have pointer := congrArg BitVec.toNat (actual_pointer t laws c n cap positive)
  have span := (window t laws c n cap).spanFits
  have fits : (window t laws c n cap).base < 2^64 := by
    change (window t laws c n cap).base + 4*n ≤ 2^64 at span
    omega
  change (BitVec.ofNat 64 ((window t laws c n cap).base + 4*0)).toNat = t.dataPtr.toNat at pointer
  simp only [Nat.mul_zero,Nat.add_zero,BitVec.toNat_ofNat] at pointer
  rw [Nat.mod_eq_of_lt fits] at pointer
  exact pointer

theorem aligned (t : Tensor) (laws : TensorLaws t)
    (c : t.metadata.isContiguous = Bool.true) (n : Nat) (cap : n ≤ t.metadata.numel)
    (checked : 0<n → t.dataPtr.toNat % 4 = 0) : (window t laws c n cap).base % 4 = 0 := by
  by_cases zero : n=0
  · simp [window,VeriTile.Torch.FP32.DataView.base,zero]
  · rw [base_pointer t laws c n cap (by omega)]
    exact checked (by omega)

def logicalValue {M : Type} (ops : MemoryPrimitives M Word) (s : M)
    (t : Tensor) (laws : TensorLaws t) (i : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue (ops.read s .fp32) t laws i

theorem direct_read {M : Type} (ops : MemoryPrimitives M Word) (s : M)
    (t : Tensor) (laws : TensorLaws t) (c : t.metadata.isContiguous = Bool.true)
    (ng : t.metadata.isNeg = Bool.false) (n : Nat) (cap : n ≤ t.metadata.numel)
    (i : Nat) (hi : i<n) :
    ops.read s .fp32 (address (window t laws c n cap) i) = logicalValue ops s t laws i := by
  rw [window_address t laws c n cap i hi]
  exact VeriTile.Torch.FP32.direct_read_eq_logical (ops.read s .fp32) t laws c ng i (by omega)

def layout (x out : Tensor) (lx : TensorLaws x) (lo : TensorLaws out)
    (h : InputContract x.metadata out.metadata) : Layout x.metadata.numel where
  window
    | .x => window x lx h.contiguous.1 x.metadata.numel (Nat.le_refl _)
    | .out => window out lo h.contiguous.2 x.metadata.numel h.capacity
  capacity := by intro r; cases r <;> exact Nat.le_refl _
  aligned := by
    intro r; cases r
    · exact aligned x lx h.contiguous.1 _ (Nat.le_refl _) (fun p => (h.alignment p).1)
    · exact aligned out lo h.contiguous.2 _ h.capacity (fun p => (h.alignment p).2)
  compatible := by
    by_cases zero : x.metadata.numel=0
    · left; simp [window,VeriTile.Torch.FP32.DataView.base,zero]
    · have positive : 0<x.metadata.numel := by omega
      have bp := base_pointer x lx h.contiguous.1 _ (Nat.le_refl _) positive
      have bo := base_pointer out lo h.contiguous.2 _ h.capacity positive
      rcases h.overlap positive with eq | sep | sep
      · left; rw [bp,bo]; exact congrArg BitVec.toNat eq
      · right; left
        change (window x lx h.contiguous.1 _ (Nat.le_refl _)).base + 4*x.metadata.numel ≤
          (window out lo h.contiguous.2 _ h.capacity).base
        rw [bp,bo]; exact sep
      · right; right
        change (window out lo h.contiguous.2 _ h.capacity).base + 4*x.metadata.numel ≤
          (window x lx h.contiguous.1 _ (Nat.le_refl _)).base
        rw [bp,bo]; exact sep

/-- Successful ordinary launch on x.device. Memory permissions are separate. -/
structure Runtime {M : Type} (ops : MemoryPrimitives M Word) (x : Tensor) : Prop where
  device : ops.launchDevice = x.metadata.device.index

/-- The first Tensor component records the object mutated, not a Python return
value. Unit records Python None. No allocation or input normalization occurs. -/
def runWrapper {M : Type} (ops : MemoryPrimitives M Word) (nativeSin : Word → Word)
    (x out : Tensor) (lx : TensorLaws x) (lo : TensorLaws out)
    (h : InputContract x.metadata out.metadata) (programs : List Nat) (before : M) :
    Except InputError (Option (Tensor × Unit × M)) := do
  checkInputs x.metadata out.metadata
  if x.metadata.numel=0 then return some (out,(),before)
  return (runJobs ops nativeSin (FP.Value.zero .fp32) (layout x out lx lo h) programs before).map
    (fun after => (out,(),after))

theorem wrapper_correct {M : Type} (ops : MemoryPrimitives M Word) (nativeSin : Word → Word)
    (x out : Tensor) (lx : TensorLaws x) (lo : TensorLaws out)
    (h : InputContract x.metadata out.metadata)
    (memory : LocalMemory ops (layout x out lx lo h))
    (schedule : Index.Schedule x.metadata.numel) (before : M) :
    ∃ after,runWrapper ops nativeSin x out lx lo h schedule.programs before = .ok (some (out,(),after)) ∧
      (∀ i,i<x.metadata.numel → logicalValue ops after out lo i = nativeSin (logicalValue ops before x lx i)) ∧
      Preserved ops (view (window out lo h.contiguous.2 x.metadata.numel h.capacity) 0) after before := by
  have accepted : checkInputs x.metadata out.metadata = .ok () := (checkTensorInputs_iff _ _).mpr h
  by_cases empty : x.metadata.numel=0
  · exact ⟨before,by simp [runWrapper,accepted,empty],by simp [empty],preserved_refl _ _ _⟩
  · obtain ⟨after,he,hv,hf,_⟩ := launch_correct ops nativeSin (FP.Value.zero .fp32)
      (layout x out lx lo h) memory schedule before
    refine ⟨after,by simp [runWrapper,accepted,empty,he],?_,hf⟩
    intro i hi
    rw [←direct_read ops after out lo h.contiguous.2 h.noNeg.2 _ h.capacity i hi]
    change read ops (layout x out lx lo h) after .out i = _
    rw [hv i hi]
    exact congrArg nativeSin (direct_read ops before x lx h.contiguous.1 h.noNeg.1 _ (Nat.le_refl _) i hi)

#axiomsClean wrapper_correct
#axiomsClean actual_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Binding
