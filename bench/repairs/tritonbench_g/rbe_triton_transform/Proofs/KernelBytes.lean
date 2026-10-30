import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelNative
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Bytes
open VeriTile Triton FP.Structural Binding Ordered
noncomputable section
open Classical
set_option maxHeartbeats 6000000
abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte
def encode (x : Word) (byte : Fin 4) : Byte := x.bits.extractLsb' (8*byte.val) 8

def decode (bytes : Fin 4 → Byte) : Word := ⟨bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0⟩

theorem decode_encode (x : Word) : decode (encode x)=x := by
  have ext : ∀ a b : Word,a.bits=b.bits → a=b := by
    rintro ⟨a⟩ ⟨b⟩ equal
    cases equal
    rfl
  apply ext
  change (x.bits.extractLsb' 24 8 ++ x.bits.extractLsb' 16 8 ++ x.bits.extractLsb' 8 8 ++ x.bits.extractLsb' 0 8)=x.bits
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext byte
  fin_cases byte
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def load (memory : Memory) (address : Nat) : Word := decode (fun i => memory (address+i.val))

def initial (x y : Tensor) (memory : Memory) : State Tagged :=
  Binding.initial x y (fun address => leaf (load memory address))

def nativeResult (p : Primitives) (r : Request) (x : Tensor) (memory : Memory) (k : Nat) : Word :=
  Ordered.result p r (x.read (load memory)) (batchIndex r k) (rowIndex r k) (columnIndex r k)

/-- Global physical byte update of the independent native result. -/
def reference (y : Tensor) (memory : Memory) (values : Nat → Word) : Memory :=
  fun address => if footprint y address then
    encode (values ((address-y.view.base)/4)) ⟨(address-y.view.base)%4,by omega⟩
  else memory address

/-- Interpret every source output at the actual output allocation. -/
def project (y : Tensor) (memory : Memory) (s : State Tagged) : Memory :=
  reference y memory (fun i => ((s.mem "Y" i).read .fp32).word)

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

theorem entry_project (x y : Tensor) (memory : Memory) : project y memory (initial x y memory)=memory := by
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

/-- Exact little-endian four-byte store. -/
def store (memory : Memory) (address : Nat) (value : Word) : Memory :=
  fun a => if h : address≤a ∧ a<address+4 then encode value ⟨a-address,by omega⟩ else memory a

theorem write_commutes (y : Tensor) (memory : Memory) (s : State Tagged)
    (k : Nat) (active : k<y.size) (v : Tagged) :
    project y memory (s.write "Y" k (.mk .fp32 v))=
      store (project y memory s) (y.view.address k) v.word := by
  funext address
  by_cases inside : y.view.address k≤address ∧ address<y.view.address k+4
  · have footprint : Binding.footprint y address := by
      unfold Binding.footprint Torch.FP32.DataView.address at *
      omega
    have div : (address-y.view.base)/4=k := by unfold Torch.FP32.DataView.address at inside; omega
    simp only [project,reference,footprint,if_true,State.write,div,
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
      simp [State.write,different activeAddress]
    · rfl

theorem logical_value (p : Primitives) (r : Request) (x y : Tensor) (input : Input x r)
    (memory : Memory) (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    (scalarResult (operations p) r (fun o => ((initial x y memory).mem "X" o).read .fp32) b m c).word=
      Ordered.result p r (x.read (load memory)) b m c := by
  unfold initial
  rw [Binding.scalar_logical (operations p) r x y input (fun a => leaf (load memory a)) b m c hb hm hc,
    source_profile]
  rfl

theorem project_reference (y : Tensor) (memory : Memory) (s : State Tagged) (values : Nat → Word)
    (all : ∀ k,k<y.size → ((s.mem "Y" k).read .fp32).word=values k) :
    project y memory s=reference y memory values := by
  funext address
  simp only [project,reference]
  split
  · rename_i active
    rw [all _ (by unfold footprint at active; omega)]
  · rfl

#guard_msgs (drop info) in
#specNonCircular nativeResult avoiding [Source.source,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Bytes
