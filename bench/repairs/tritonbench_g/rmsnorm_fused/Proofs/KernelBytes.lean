import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelNative
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RmsnormFused.Bytes
open VeriTile Triton FP.Structural Binding Ordered
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte
def encode (x : Word) (byte : Fin 4) : Byte := x.extractLsb' (8*byte.val) 8

def decode (bytes : Fin 4 → Byte) : Word := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0

theorem decode_encode (x : Word) : decode (encode x)=x := by
  change (x.extractLsb' 24 8 ++ x.extractLsb' 16 8 ++ x.extractLsb' 8 8 ++ x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext byte
  fin_cases byte
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def load (memory : Memory) (address : Nat) : Word := decode (fun i => memory (address+i.val))

def initial (x w y : Tensor) (memory : Memory) : State (Term Word) :=
  Binding.initial x w y (fun address => .input (load memory address))

def nativeResult (I : Instructions) (P : Policy) (R : Reductions)
    (x w : Tensor) (memory : Memory) (N B : Nat) (epsilon : Word) (k : Nat) : Word :=
  Ordered.reference I P R N B epsilon
    (fun j => x.read (load memory) ((k/N)*N+j)) (w.read (load memory)) (k%N)

/-- Global physical byte update of the independent native result. -/
def reference (y : Tensor) (memory : Memory) (values : Nat → Word) : Memory :=
  fun address => if footprint y address then
    encode (values ((address-y.view.base)/4)) ⟨(address-y.view.base)%4,by omega⟩
  else memory address

/-- Interpret every source output word at its actual physical address. -/
def project (I : Instructions) (P : Policy) (R : Reductions) (N : Nat)
    (y : Tensor) (memory : Memory) (state : State (Term Word)) : Memory :=
  reference y memory (fun i => interpret I P R (i%N) (Runtime.read state "Y" i))

theorem reference_byte (y : Tensor) (memory : Memory) (values : Nat → Word)
    (k : Nat) (active : k<y.size) (i : Fin 4) :
    reference y memory values (y.view.address k+i.val)=encode (values k) i := by
  have div : (y.view.base+4*k+i.val-y.view.base)/4=k := by omega
  have rem : (y.view.base+4*k+i.val-y.view.base)%4=i.val := by omega
  simp only [reference,footprint,Torch.FP32.DataView.address]
  rw [if_pos (by omega),div]
  congr 1
  exact Fin.ext rem

theorem reference_load (y : Tensor) (memory : Memory) (values : Nat → Word)
    (k : Nat) (active : k<y.size) :
    load (reference y memory values) (y.view.address k)=values k := by
  unfold load
  have bytes : (fun i : Fin 4 => reference y memory values (y.view.address k+i.val))=encode (values k) := by
    funext i
    exact reference_byte y memory values k active i
  rw [bytes,decode_encode]

theorem reference_frame (y : Tensor) (memory : Memory) (values : Nat → Word)
    (address : Nat) (outside : ¬footprint y address) :
    reference y memory values address=memory address := by simp [reference,outside]

theorem entry_project (I : Instructions) (P : Policy) (R : Reductions) (N : Nat)
    (x w y : Tensor) (memory : Memory) : project I P R N y memory (initial x w y memory)=memory := by
  funext address
  unfold project reference
  split
  · rename_i inside
    change encode (load memory (y.view.address ((address-y.view.base)/4)))
      ⟨(address-y.view.base)%4,by omega⟩=memory address
    have original := congrFun (encode_decode (fun i : Fin 4 =>
      memory (y.view.address ((address-y.view.base)/4)+i.val))) ⟨(address-y.view.base)%4,by omega⟩
    unfold load
    rw [original]
    congr 1
    unfold footprint at inside
    unfold Torch.FP32.DataView.address
    simp only [Fin.val_mk]
    omega
  · rfl

/-- One exact four-byte native store, independent of the symbolic interpreter. -/
def store (memory : Memory) (address : Nat) (value : Word) : Memory :=
  fun a => if h : address≤a ∧ a<address+4 then encode value ⟨a-address,by omega⟩ else memory a

/-- Source writes commute with actual little-endian stores at the returned
Tensor's pointer. This proves the physical write connection locally. -/
theorem write_commutes (I : Instructions) (P : Policy) (R : Reductions) (N : Nat)
    (y : Tensor) (memory : Memory) (s : State (Term Word)) (k : Nat) (active : k<y.size)
    (v : Term Word) :
    project I P R N y memory (s.write "Y" k (.mk .real v))=
      store (project I P R N y memory s) (y.view.address k) (interpret I P R (k%N) v) := by
  funext address
  by_cases inside : y.view.address k≤address ∧ address<y.view.address k+4
  · have footprint : Binding.footprint y address := by
      unfold Binding.footprint Torch.FP32.DataView.address at *
      omega
    have div : (address-y.view.base)/4=k := by unfold Torch.FP32.DataView.address at inside; omega
    simp only [project,reference,footprint,if_true,Runtime.read,State.write,div,
      true_and,if_true,Cell.read_same,store,inside]
    congr 1
    apply Fin.ext
    unfold Torch.FP32.DataView.address at inside ⊢
    simp only [Fin.val_mk]
    omega
  · have different : footprint y address → (address-y.view.base)/4≠k := by
      unfold footprint Torch.FP32.DataView.address at *
      omega
    simp only [project,reference,store,inside,if_false]
    split
    · rename_i activeAddress
      simp [Runtime.read,State.write,different activeAddress]
    · rfl

theorem logical_value (I : Instructions) (P : Policy) (R : Reductions)
    (x w y : Tensor) (memory : Memory) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (epsilon : Word) (row col : Nat)
    (hr : row<(geometry x.info).rows) (hc : col<width x.info) :
    interpret I P R col (Runtime.value graphAlgebra (initial x w y memory) "X" "W"
      row (width x.info) (RmsnormFused.block (width x.info)) (.input epsilon) col)=
      nativeResult I P R x w memory (width x.info) (RmsnormFused.block (width x.info)) epsilon
        (row*width x.info+col) := by
  let sourceWords := fun region offset =>
    load memory ((if region=("X" : RegionName) then x.view else if region="W" then w.view else y.view).address offset)
  rw [source_lowering I P R sourceWords _ _ row col epsilon (initial x w y memory)
    (fun _ => rfl) (fun _ => rfl) hc]
  have size := input_sizes x w eps c
  have coordinates := offset_coordinates (geometry x.info) (geometry_valid x.info w.info eps c) row col hc
  simp only [Shape.offset,geometry] at coordinates
  unfold nativeResult
  rw [coordinates.1,coordinates.2]
  -- Only logical active inputs are read by the independent recurrence.
  unfold Ordered.reference
  have inputs : (fun j : Fin (RmsnormFused.block (width x.info)) =>
      if j.val<width x.info then convert I P (atom (sourceWords "X" (row*width x.info+j.val))) else literal I 0)=
      (fun j : Fin (RmsnormFused.block (width x.info)) =>
      if j.val<width x.info then convert I P (atom (x.read (load memory) (row*width x.info+j.val))) else literal I 0) := by
    funext j
    split
    · rename_i hj
      rw [x.read_view _ _ (by rw [size.1]; exact offset_bounded (geometry x.info) row j.val hr hj)]
      rfl
    · rfl
  rw [inputs]
  dsimp only
  rw [x.read_view (load memory) (row*width x.info+col) (by rw [size.1]; exact offset_bounded (geometry x.info) row col hr hc),
    w.read_view (load memory) col (by omega)]
  rfl

#guard_msgs (drop info) in
#specNonCircular nativeResult avoiding [lower,interpret,Runtime.value,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Bytes
