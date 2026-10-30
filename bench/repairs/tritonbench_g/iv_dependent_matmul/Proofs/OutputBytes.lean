import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Execution

namespace VeriTile.Bench.Repairs.IvDependentMatmul.OutputBytes
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000
abbrev Read := Nat → BitVec 8
structure View where
  region : RegionName
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

/-- A representation of C's current words in a common physical byte reader.
The initial identity and single-store commuting theorem below justify its
composition through the actual source's complete sequence of stores. -/
def render (original : Read) (c : View) (s : State Word) : Read := fun a =>
  if c.base≤a ∧ a<c.base+4*c.extent then
    byte (((s.mem c.region ((a-c.base)/4)).read .fp32).bits)
      ((a-c.base)%4)
  else original a

def Bound (read : Read) (c : View) (s : State Word) : Prop :=
  ∀ i,i<c.extent → s.mem c.region i=Cell.mk .fp32 ⟨read32 read (c.base+4*i)⟩

theorem render_at (read : Read) (c : View) (s : State Word)
    (i j : Nat) (hi : i<c.extent) (hj : j<4) :
    render read c s (c.base+4*i+j)=
      byte (((s.mem c.region i).read .fp32).bits) j := by
  have inside : c.base≤c.base+4*i+j ∧
      c.base+4*i+j<c.base+4*c.extent := by omega
  have hd : (c.base+4*i+j-c.base)/4=i := by omega
  have hm : (c.base+4*i+j-c.base)%4=j := by omega
  simp only [render,if_pos inside,hd,hm]

theorem render_read (read : Read) (c : View) (s : State Word)
    (i : Nat) (hi : i<c.extent) :
    read32 (render read c s) (c.base+4*i)=
      ((s.mem c.region i).read .fp32).bits := by
  unfold read32
  have h0 := render_at read c s i 0 hi (by decide)
  simp only [Nat.add_zero] at h0
  rw [h0,render_at read c s i 1 hi (by decide),render_at read c s i 2 hi (by decide),
    render_at read c s i 3 hi (by decide)]
  exact pack_bytes _

theorem render_initial (read : Read) (c : View) (s : State Word)
    (bound : Bound read c s) : render read c s=read := by
  funext a
  by_cases inside : c.base≤a ∧ a<c.base+4*c.extent
  · let i := (a-c.base)/4
    let j := (a-c.base)%4
    have hi : i<c.extent := by dsimp [i]; omega
    have hj : j<4 := Nat.mod_lt _ (by decide)
    have eq : a=c.base+4*i+j := by dsimp [i,j]; omega
    rw [eq,render_at read c s i j hi hj,bound i hi]
    simp only [Cell.read_same]
    unfold read32
    rw [unpack_byte _ _ _ _ j hj]
    interval_cases j <;> simp
  · simp only [render,if_neg inside]

theorem render_write (read : Read) (c : View) (s : State Word)
    (i : Nat) (hi : i<c.extent) (v : Word) :
    render read c (s.write c.region i (.mk .fp32 v))=
      store (render read c s) (c.base+4*i) v.bits := by
  funext a
  by_cases inside : c.base≤a ∧ a<c.base+4*c.extent
  · have index : (a-c.base)/4=i ↔
        c.base+4*i≤a ∧ a<c.base+4*i+4 := by omega
    by_cases hit : (a-c.base)/4=i
    · have remain : (a-c.base)%4=a-(c.base+4*i) := by omega
      simp only [render,if_pos inside,State.write,eq_self_iff_true,hit,and_self,ite_true,Cell.read_same,
        store,if_pos (index.mp hit),remain]
    · simp only [render,if_pos inside,State.write,eq_self_iff_true,true_and,if_neg hit,
        store,if_neg (index.not.mp hit)]
  · have miss : ¬(c.base+4*i≤a ∧ a<c.base+4*i+4) := by omega
    simp only [render,if_neg inside,store,if_neg miss]

theorem render_frame (read : Read) (c : View) (s : State Word)
    (a : Nat) (outside : a<c.base ∨ c.base+4*c.extent≤a) :
    render read c s a=read a := by
  simp only [render,if_neg (by omega : ¬(c.base≤a ∧ a<c.base+4*c.extent))]

#axiomsClean pack_bytes
#axiomsClean unpack_byte
#axiomsClean store_read
#axiomsClean render_initial
#axiomsClean render_write
#axiomsClean render_read
#axiomsClean render_frame
end
end VeriTile.Bench.Repairs.IvDependentMatmul.OutputBytes
