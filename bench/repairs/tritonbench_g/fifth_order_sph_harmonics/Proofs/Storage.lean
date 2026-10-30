import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Packing

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Storage
open VeriTile.Triton FP.Structural Packing
noncomputable section
abbrev Heap := Nat → BitVec 8

def read (heap : Heap) (address : Nat) : Word := fun k=>heap (address+k.val)

/-- Four byte updates at the actual address; all other byte values survive. -/
def write (heap : Heap) (address : Nat) (value : Word) : Heap := fun q=>
  if inside : address≤q ∧ q<address+4 then value ⟨q-address,by omega⟩ else heap q

theorem write_byte (heap : Heap) (address : Nat) (value : Word) (k : Fin 4) :
    write heap address value (address+k.val)=value k := by
  have hk:=k.isLt
  simp [write,show address≤address+k.val ∧ address+k.val<address+4 by omega]

theorem write_read (heap : Heap) (address : Nat) (value : Word) :
    read (write heap address value) address=value := by funext k;exact write_byte heap address value k

theorem write_frame (heap : Heap) (address : Nat) (value : Word) (q : Nat)
    (outside : q<address ∨ address+4≤q) : write heap address value q=heap q := by
  simp [write,show ¬(address≤q ∧ q<address+4) by omega]

/-- An image of every destination cell, including arbitrary initial padding.
The replay theorem below shows that this image commutes with each actual store. -/
def image (region : RegionName) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → Cell Word) : Heap := fun q=>
  if base≤q ∧ q<base+4*extent then
    ((memory region ((q-base)/4)).read .real) ⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
  else before q

theorem image_byte (region : RegionName) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → Cell Word) (i : Nat) (hi : i<extent) (k : Fin 4) :
    image region base extent before memory (base+4*i+k.val)=((memory region i).read .real) k := by
  have hk:=k.isLt
  have inside : base≤base+4*i+k.val ∧ base+4*i+k.val<base+4*extent := by omega
  have div : (base+4*i+k.val-base)/4=i := by omega
  have mod : (base+4*i+k.val-base)%4=k.val := by omega
  simp [image,inside,div,mod]

theorem image_frame (region : RegionName) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → Cell Word) (q : Nat)
    (outside : q<base ∨ base+4*extent≤q) : image region base extent before memory q=before q := by
  simp [image,show ¬(base≤q ∧ q<base+4*extent) by omega]

theorem image_initial (region : RegionName) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → Cell Word)
    (backed : ∀ i,i<extent → (memory region i).read .real=read before (base+4*i)) :
    image region base extent before memory=before := by
  funext q
  by_cases inside : base≤q ∧ q<base+4*extent
  · let i:=(q-base)/4
    let k:Fin 4:=⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
    have hi:i<extent := by dsimp [i];omega
    have eq:base+4*i+k.val=q := by dsimp [i,k];omega
    rw [←eq,image_byte _ _ _ _ _ i hi k,backed i hi]
    rfl
  · exact image_frame _ _ _ _ _ q (by omega)

theorem image_write (region : RegionName) (base extent : Nat) (before : Heap)
    (state : State Word) (i : Nat) (hi : i<extent) (value : Word) :
    image region base extent before (state.write region i (.mk .real value)).mem=
      write (image region base extent before state.mem) (base+4*i) value := by
  funext q
  by_cases hit : base+4*i≤q ∧ q<base+4*i+4
  · let k:Fin 4:=⟨q-(base+4*i),by omega⟩
    have eq:base+4*i+k.val=q := by dsimp [k];omega
    rw [←eq,image_byte _ _ _ _ _ i hi k,write_byte]
    simp only [State.write,eq_self_iff_true,and_self,ite_true,Cell.read_same]
  · rw [write_frame _ _ _ _ (by omega)]
    by_cases inside : base≤q ∧ q<base+4*extent
    · have neq:(q-base)/4≠i := by omega
      simp [image,inside,State.write,neq]
    · simp [image,inside]

/-- The complete byte writer is projected from the original store fold. -/
theorem fold_projection {ι : Type} (events : List ι) (region : RegionName) (base extent : Nat)
    (active : ι → Prop) [DecidablePred active] (index : ι → Nat) (value : ι → Word)
    (bounded : ∀ e∈events,active e → index e<extent) (before : Heap) (state : State Word) :
    image region base extent before
      ((events.foldl (fun s e=>if active e then s.write region (index e) (.mk .real (value e)) else s) state).mem)=
      events.foldl (fun heap e=>if active e then write heap (base+4*index e) (value e) else heap)
        (image region base extent before state.mem) := by
  induction events generalizing state with
  | nil=>rfl
  | cons e tail ih=>
    simp only [List.foldl_cons]
    by_cases enabled:active e
    · simp only [if_pos enabled]
      rw [ih (fun a ha=>bounded a (by simp [ha])),image_write _ _ _ _ _ _ (bounded e (by simp) enabled)]
    · simp only [if_neg enabled]
      exact ih (fun a ha=>bounded a (by simp [ha])) state

#axiomsClean write_read
#axiomsClean write_frame
#axiomsClean image_initial
#axiomsClean image_write
#axiomsClean fold_projection
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Storage
