import bench.repairs.tritonbench_g.streamk_matmul.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.StreamK.Checked.ByteMemory
open TensorBinding VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
abbrev Read := ByteRead
structure View where
  base : Nat
  extent : Nat

def read32 (read : Read) (address : Nat) : BitVec 32 :=
  BitVec.ofNat 32 ((read address).toNat+256*(read (address+1)).toNat+
    65536*(read (address+2)).toNat+16777216*(read (address+3)).toNat)

def byte (w : BitVec 32) (j : Nat) : BitVec 8 := w.extractLsb' (8*j) 8

theorem pack_bytes (w : BitVec 32) :
    BitVec.ofNat 32 ((byte w 0).toNat+256*(byte w 1).toNat+
      65536*(byte w 2).toNat+16777216*(byte w 3).toNat)=w := by
  apply BitVec.eq_of_toNat_eq
  have hw := w.isLt
  simp only [byte,BitVec.extractLsb'_toNat,Nat.shiftRight_eq_div_pow,BitVec.toNat_ofNat] at *
  norm_num at *
  omega

theorem unpack_byte (b0 b1 b2 b3 : BitVec 8) (j : Nat) (hj : j<4) :
    byte (BitVec.ofNat 32 (b0.toNat+256*b1.toNat+65536*b2.toNat+16777216*b3.toNat)) j=
      if j=0 then b0 else if j=1 then b1 else if j=2 then b2 else b3 := by
  have h0 := b0.isLt
  have h1 := b1.isLt
  have h2 := b2.isLt
  have h3 := b3.isLt
  interval_cases j <;> apply BitVec.eq_of_toNat_eq <;>
    simp only [byte,BitVec.extractLsb'_toNat,Nat.shiftRight_eq_div_pow,BitVec.toNat_ofNat] <;>
    norm_num at * <;> omega

/-- Four little-endian byte writes at the actual output pointer. -/
def store (read : Read) (ptr : Nat) (w : BitVec 32) : Read := fun a =>
  if ptr≤a ∧ a<ptr+4 then byte w (a-ptr) else read a

theorem store_read (read : Read) (ptr : Nat) (w : BitVec 32) : read32 (store read ptr w) ptr=w := by
  simp only [read32,store]
  simp only [Nat.le_refl,Nat.lt_add_of_pos_right (by decide : 0<4),and_self,ite_true,Nat.sub_self]
  have h1 : ptr≤ptr+1 ∧ ptr+1<ptr+4 := by omega
  have h2 : ptr≤ptr+2 ∧ ptr+2<ptr+4 := by omega
  have h3 : ptr≤ptr+3 ∧ ptr+3<ptr+4 := by omega
  simp only [if_pos h1,if_pos h2,if_pos h3,Nat.add_sub_cancel_left]
  exact pack_bytes w

def render (original : Read) (c : View) (values : Nat → Word) : Read := fun a =>
  if c.base≤a ∧ a<c.base+4*c.extent then
    byte (values ((a-c.base)/4)).bits ((a-c.base)%4) else original a

def Bound (read : Read) (c : View) (values : Nat → Word) : Prop :=
  ∀ i,i<c.extent → (values i).bits=read32 read (c.base+4*i)

theorem render_at (read : Read) (c : View) (values : Nat → Word)
    (i j : Nat) (hi : i<c.extent) (hj : j<4) :
    render read c values (c.base+4*i+j)=byte (values i).bits j := by
  have inside : c.base≤c.base+4*i+j ∧ c.base+4*i+j<c.base+4*c.extent := by omega
  have hd : (c.base+4*i+j-c.base)/4=i := by omega
  have hm : (c.base+4*i+j-c.base)%4=j := by omega
  simp only [render,if_pos inside,hd,hm]

theorem render_read (read : Read) (c : View) (values : Nat → Word) (i : Nat) (hi : i<c.extent) :
    read32 (render read c values) (c.base+4*i)=(values i).bits := by
  unfold read32
  have h0 := render_at read c values i 0 hi (by decide)
  simp only [Nat.add_zero] at h0
  rw [h0,render_at read c values i 1 hi (by decide),render_at read c values i 2 hi (by decide),
    render_at read c values i 3 hi (by decide)]
  exact pack_bytes _

theorem render_initial (read : Read) (c : View) (values : Nat → Word)
    (bound : Bound read c values) : render read c values=read := by
  funext a
  by_cases inside : c.base≤a ∧ a<c.base+4*c.extent
  · let i := (a-c.base)/4
    let j := (a-c.base)%4
    have hi : i<c.extent := by dsimp [i]; omega
    have hj : j<4 := Nat.mod_lt _ (by decide)
    have eq : a=c.base+4*i+j := by dsimp [i,j]; omega
    rw [eq,render_at read c values i j hi hj,bound i hi]
    unfold read32
    rw [unpack_byte _ _ _ _ j hj]
    interval_cases j <;> simp
  · simp only [render,if_neg inside]

theorem render_write (read : Read) (c : View) (values : Nat → Word)
    (i : Nat) (hi : i<c.extent) (v : Word) :
    render read c (Function.update values i v)=store (render read c values) (c.base+4*i) v.bits := by
  funext a
  by_cases inside : c.base≤a ∧ a<c.base+4*c.extent
  · have index : (a-c.base)/4=i ↔ c.base+4*i≤a ∧ a<c.base+4*i+4 := by omega
    by_cases hit : (a-c.base)/4=i
    · have remain : (a-c.base)%4=a-(c.base+4*i) := by omega
      simp only [render,if_pos inside,hit,Function.update_self,store,if_pos (index.mp hit),remain]
      rw [hit,Function.update_self]
    · simp only [render,if_pos inside,Function.update_of_ne hit,store,if_neg (index.not.mp hit)]
  · have miss : ¬(c.base+4*i≤a ∧ a<c.base+4*i+4) := by omega
    simp only [render,if_neg inside,store,if_neg miss]

def Outside (c : View) (a : Nat) : Prop := a<c.base ∨ c.base+4*c.extent≤a

theorem render_frame (read : Read) (c : View) (values : Nat → Word)
    (a : Nat) (outside : Outside c a) : render read c values a=read a := by
  have miss : ¬(c.base≤a ∧ a<c.base+4*c.extent) := by
    unfold Outside at outside; omega
  simp only [render,if_neg miss]

def Separate (a b : View) : Prop := a.base+4*a.extent≤b.base ∨ b.base+4*b.extent≤a.base

theorem render_store_other (read : Read) (a b : View) (values : Nat → Word)
    (i : Nat) (hi : i<b.extent) (w : Word) (separate : Separate a b) :
    render (store read (b.base+4*i) w.bits) a values=
      store (render read a values) (b.base+4*i) w.bits := by
  funext address
  by_cases ha : a.base≤address ∧ address<a.base+4*a.extent
  · have miss : ¬(b.base+4*i≤address ∧ address<b.base+4*i+4) := by
      unfold Separate at separate; omega
    simp only [render,if_pos ha,store,if_neg miss]
  · simp only [render,if_neg ha,store]

theorem render_read_other (read : Read) (a b : View) (values : Nat → Word)
    (i : Nat) (hi : i<b.extent) (separate : Separate a b) :
    read32 (render read a values) (b.base+4*i)=read32 read (b.base+4*i) := by
  have out : ∀ j,j<4 → Outside a (b.base+4*i+j) := by
    intro j hj; unfold Outside Separate at *; omega
  unfold read32
  have h0 := render_frame read a values (b.base+4*i+0) (out 0 (by decide))
  simp only [Nat.add_zero] at h0
  rw [h0,render_frame _ _ _ _ (out 1 (by decide)),render_frame _ _ _ _ (out 2 (by decide)),
    render_frame _ _ _ _ (out 3 (by decide))]

def extend (f : Fin n → Word) : Nat → Word := fun i => if h:i<n then f ⟨i,h⟩ else FP.Value.zero .fp32
@[simp] theorem extend_at (f : Fin n → Word) (i : Fin n) : extend f i.val=f i := by simp [extend]

theorem extend_update (f : Fin n → Word) (i : Fin n) (v : Word) :
    extend (Function.update f i v)=Function.update (extend f) i.val v := by
  funext j
  by_cases hj : j=i.val
  · subst j; simp
  · by_cases bound : j<n
    · have ne : (⟨j,bound⟩ : Fin n)≠i := by intro h; exact hj (congrArg Fin.val h)
      simp [extend,bound,Function.update_of_ne hj,Function.update_of_ne ne]
    · simp [extend,bound,Function.update_of_ne hj]

#axiomsClean render_initial
#axiomsClean render_write
#axiomsClean render_store_other
#axiomsClean render_read_other
#axiomsClean extend_update
end
end VeriTile.Bench.Repairs.StreamK.Checked.ByteMemory
