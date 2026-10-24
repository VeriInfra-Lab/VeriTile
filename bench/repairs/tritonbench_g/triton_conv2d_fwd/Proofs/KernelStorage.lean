import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNative
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumericProof
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Physical
namespace VeriTile.Bench.Repairs.Conv2d.KernelStorage
open VeriTile.Triton FP.Structural Native KernelBinding KernelGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16384
abbrev Value := KernelNumeric.Value Word
abbrev Operations := KernelNumeric.Operations Word
abbrev Heap := Physical.Heap

def read (heap : Heap) (address : Nat) : Word := fun k=>heap (address+k.val)

/-- Physical observation of the destination region. Initial-image and local
store-commutation theorems below establish concrete evolving byte writes; this
is not an assumed final-byteheap agreement or an output-valued contract. -/
def image (o : Operations) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → FP.Structural.Cell Value) : Heap := fun q=>
  if base≤q ∧ q<base+4*extent then
    KernelNumeric.scalar o ((memory "output" ((q-base)/4)).read .fp32)
      ⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
  else before q

theorem image_byte (o : Operations) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → FP.Structural.Cell Value) (i : Nat) (hi : i<extent) (k : Fin 4) :
    image o base extent before memory (base+4*i+k.val)=
      KernelNumeric.scalar o ((memory "output" i).read .fp32) k := by
  have hk:=k.isLt
  have inside : base≤base+4*i+k.val ∧ base+4*i+k.val<base+4*extent := by omega
  have hd : (base+4*i+k.val-base)/4=i := by omega
  have hm : (base+4*i+k.val-base)%4=k.val := by omega
  simp [image,inside,hd,hm]

theorem image_frame (o : Operations) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → FP.Structural.Cell Value) (q : Nat)
    (outside : q<base ∨ base+4*extent≤q) : image o base extent before memory q=before q := by
  simp [image,show ¬(base≤q ∧ q<base+4*extent) by omega]

theorem image_initial (o : Operations) (base extent : Nat) (before : Heap)
    (memory : RegionName → Nat → FP.Structural.Cell Value)
    (backed : ∀ i,i<extent → (memory "output" i).read .fp32=.scalar (read before (base+4*i))) :
    image o base extent before memory=before := by
  funext q
  by_cases inside : base≤q ∧ q<base+4*extent
  · let i:=(q-base)/4
    let k:Fin 4:=⟨(q-base)%4,Nat.mod_lt _ (by omega)⟩
    have hi:i<extent := by dsimp [i];omega
    have eq:base+4*i+k.val=q := by dsimp [i,k];omega
    rw [←eq,image_byte o _ _ _ _ i hi k,backed i hi]
    rfl
  · exact image_frame _ _ _ _ _ q (by omega)

/-- Every actual typed store commutes with four concrete byte writes. -/
theorem image_write (o : Operations) (base extent : Nat) (before : Heap) (state : State Value)
    (i : Nat) (hi : i<extent) (value : Value) :
    image o base extent before (state.write "output" i (.mk .fp32 value)).mem=
      Physical.write (image o base extent before state.mem) (base+4*i) (KernelNumeric.scalar o value) := by
  funext q
  by_cases hit : base+4*i≤q ∧ q<base+4*i+4
  · let k:Fin 4:=⟨q-(base+4*i),by omega⟩
    have eq:base+4*i+k.val=q := by dsimp [k];omega
    rw [←eq,image_byte o _ _ _ _ i hi k,Physical.write_byte]
    simp only [State.write,eq_self_iff_true,and_self,ite_true,FP.Structural.Cell.read_same]
  · rw [Physical.write_frame _ _ _ _ (by omega)]
    by_cases inside : base≤q ∧ q<base+4*extent
    · have neq:(q-base)/4≠i := by omega
      simp [image,inside,State.write,neq]
    · simp [image,inside]

theorem fold_projection {ι : Type} (events : List ι) (o : Operations) (base extent : Nat)
    (active : ι → Prop) [DecidablePred active] (index : ι → Nat) (value : ι → Value)
    (bounded : ∀ e∈events,active e → index e<extent) (before : Heap) (state : State Value) :
    image o base extent before
      ((events.foldl (fun s e=>if active e then s.write "output" (index e) (.mk .fp32 (value e)) else s) state).mem)=
      events.foldl (fun heap e=>if active e then
        Physical.write heap (base+4*index e) (KernelNumeric.scalar o (value e)) else heap)
        (image o base extent before state.mem) := by
  induction events generalizing state with
  | nil=>rfl
  | cons e tail ih=>
    simp only [List.foldl_cons]
    by_cases enabled:active e
    · simp only [if_pos enabled]
      rw [ih (fun a ha=>bounded a (by simp [ha])),image_write o _ _ _ _ _ (bounded e (by simp) enabled)]
    · simp only [if_neg enabled]
      exact ih (fun a ha=>bounded a (by simp [ha])) state

#axiomsClean image_initial
#axiomsClean image_write
#axiomsClean fold_projection
end
end VeriTile.Bench.Repairs.Conv2d.KernelStorage
