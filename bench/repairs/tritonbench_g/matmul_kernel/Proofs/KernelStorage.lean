import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelReference
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Physical
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelStorage
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
abbrev Word := FP.Value .fp32
abbrev Heap := Nat → BitVec 8
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def read (heap : Heap) (address : Nat) : Word := ⟨Physical.decode (fun k=>heap (address+k.val))⟩
def bytes (value : Word) : Fin 4 → BitVec 8 := Physical.encode value.bits

theorem bytes_read (heap : Heap) (address : Nat) (k : Fin 4) :
    bytes (read heap address) k=heap (address+k.val) :=
  congrFun (Physical.encode_decode (fun k=>heap (address+k.val))) k

def write (heap : Heap) (address : Nat) (value : Word) : Heap := fun q=>
  if inside : address≤q ∧ q<address+4 then bytes value ⟨q-address,by omega⟩ else heap q

theorem write_byte (heap : Heap) (address : Nat) (value : Word) (k : Fin 4) :
    write heap address value (address+k.val)=bytes value k := by
  have hk:=k.isLt
  simp [write,show address≤address+k.val ∧ address+k.val<address+4 by omega]

theorem write_frame (heap : Heap) (address : Nat) (value : Word) (q : Nat)
    (outside : q<address ∨ address+4≤q) : write heap address value q=heap q := by
  simp [write,show ¬(address≤q ∧ q<address+4) by omega]

/-- Physical interpretation of all destination cells, including untouched
padding. The initial-image and store-commutation proofs make this an actual
byte-store interpretation of the source state, not a result specification. -/
def image (base extent : Nat) (before : Heap) (memory : RegionName → Nat → Cell Word) : Heap := fun q=>
  if base≤q ∧ q<base+4*extent then
    bytes ((memory "C" ((q-base)/4)).read .real) ⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
  else before q

theorem image_byte (base extent : Nat) (before : Heap) (memory : RegionName → Nat → Cell Word)
    (i : Nat) (hi : i<extent) (k : Fin 4) :
    image base extent before memory (base+4*i+k.val)=bytes ((memory "C" i).read .real) k := by
  have hk:=k.isLt
  have inside : base≤base+4*i+k.val ∧ base+4*i+k.val<base+4*extent := by omega
  have div : (base+4*i+k.val-base)/4=i := by omega
  have mod : (base+4*i+k.val-base)%4=k.val := by omega
  simp [image,inside,div,mod]

theorem image_frame (base extent : Nat) (before : Heap) (memory : RegionName → Nat → Cell Word)
    (q : Nat) (outside : q<base ∨ base+4*extent≤q) : image base extent before memory q=before q := by
  simp [image,show ¬(base≤q ∧ q<base+4*extent) by omega]

theorem image_initial (base extent : Nat) (before : Heap) (memory : RegionName → Nat → Cell Word)
    (backed : ∀ i,i<extent → (memory "C" i).read .real=read before (base+4*i)) :
    image base extent before memory=before := by
  funext q
  by_cases inside : base≤q ∧ q<base+4*extent
  · let i:=(q-base)/4
    let k:Fin 4:=⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
    have hi:i<extent := by dsimp [i];omega
    have eq:base+4*i+k.val=q := by dsimp [i,k];omega
    rw [←eq,image_byte _ _ _ _ i hi k,backed i hi,bytes_read]
  · exact image_frame _ _ _ _ q (by omega)

theorem image_write (base extent : Nat) (before : Heap) (state : State Word)
    (i : Nat) (hi : i<extent) (value : Word) :
    image base extent before (state.write "C" i (.mk .real value)).mem=
      write (image base extent before state.mem) (base+4*i) value := by
  funext q
  by_cases hit : base+4*i≤q ∧ q<base+4*i+4
  · let k:Fin 4:=⟨q-(base+4*i),by omega⟩
    have eq:base+4*i+k.val=q := by dsimp [k];omega
    rw [←eq,image_byte _ _ _ _ i hi k,write_byte]
    simp only [State.write,eq_self_iff_true,and_self,ite_true,Cell.read_same]
  · rw [write_frame _ _ _ _ (by omega)]
    by_cases inside : base≤q ∧ q<base+4*extent
    · have neq:(q-base)/4≠i := by omega
      simp [image,inside,State.write,neq]
    · simp [image,inside]

theorem fold_projection {ι : Type} (events : List ι) (base extent : Nat)
    (active : ι → Prop) [DecidablePred active] (index : ι → Nat) (value : ι → Word)
    (bounded : ∀ e∈events,active e → index e<extent) (before : Heap) (state : State Word) :
    image base extent before
      ((events.foldl (fun s e=>if active e then s.write "C" (index e) (.mk .real (value e)) else s) state).mem)=
      events.foldl (fun heap e=>if active e then write heap (base+4*index e) (value e) else heap)
        (image base extent before state.mem) := by
  induction events generalizing state with
  | nil=>rfl
  | cons e tail ih=>
    simp only [List.foldl_cons]
    by_cases enabled:active e
    · simp only [if_pos enabled]
      rw [ih (fun a ha=>bounded a (by simp [ha])),image_write _ _ _ _ _ (bounded e (by simp) enabled)]
    · simp only [if_neg enabled]
      exact ih (fun a ha=>bounded a (by simp [ha])) state

#axiomsClean bytes_read
#axiomsClean image_initial
#axiomsClean image_write
#axiomsClean fold_projection
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelStorage
