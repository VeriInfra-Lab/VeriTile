import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Host

namespace VeriTile.Bench.Repairs.Conv2d.Lowering
set_option maxHeartbeats 1800000
abbrev i32 (z : Int) := BitVec.ofInt 32 z

def input32 (s : Shape) (g r h w c : Nat) : BitVec 32 :=
  i32 s.sn*i32 (batch s r)+i32 s.sc*i32 g*i32 s.cg+i32 s.sc*i32 c+
    i32 s.sx*i32 (ih s r h)+i32 s.sy*i32 (iw s r w)
def weight32 (s : Shape) (g f h w c : Nat) : BitVec 32 :=
  i32 s.so*i32 f+i32 s.so*i32 g*i32 s.og+i32 s.tc*i32 c+i32 s.tx*i32 h+i32 s.ty*i32 w
def output32 (s : Shape) (x : Cell) : BitVec 32 :=
  i32 (batch s x.spatial)*i32 s.outputs*i32 s.oh*i32 s.ow+
    (i32 x.group*i32 s.og+i32 x.feature)*i32 s.oh*i32 s.ow+
    i32 (outH s x.spatial)*i32 s.ow+i32 (outW s x.spatial)

theorem input32_word (s : Shape) (g r h w c : Nat) :
    input32 s g r h w c=i32 (inputPointer s g r h w c) := by
  simp only [input32,inputPointer,i32,BitVec.ofInt_add,BitVec.ofInt_mul]
theorem weight32_word (s : Shape) (g f h w c : Nat) :
    weight32 s g f h w c=i32 (weightPointer s g f h w c) := by
  simp only [weight32,weightPointer,i32,Nat.cast_add,Nat.cast_mul,BitVec.ofInt_add,BitVec.ofInt_mul]
theorem output32_word (s : Shape) (x : Cell) :
    output32 s x=i32 (outputPointer s x) := by
  simp only [output32,outputPointer,i32,Nat.cast_add,Nat.cast_mul,BitVec.ofInt_add,BitVec.ofInt_mul]

theorem signed32_exact (z : Int) (h : Signed32 z) : (i32 z).toInt=z :=
  BitVec.toInt_ofInt_eq_self (w:=32) (by decide) (by simpa using h.1) (by simpa using h.2)

theorem signed_compare (a b : Int) (ha : Signed32 a) (hb : Signed32 b) :
    (i32 a).slt (i32 b)=decide (a<b) := by
  rw [BitVec.slt_eq_decide,signed32_exact a ha,signed32_exact b hb]

theorem widened (z : Int) (h : Signed32 z) :
    (i32 z).signExtend 64=BitVec.ofInt 64 z := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),signed32_exact z h]
  symm
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · unfold Signed32 at h;norm_num at *;omega
  · unfold Signed32 at h;norm_num at *;omega

/-- Sign-extend the formed i32 element displacement before scaling to a
64-bit byte pointer. Inactive negative offsets may form arbitrary pointers. -/
def pointer (base : Nat) (offset : BitVec 32) (k : Fin 4) : Nat :=
  (BitVec.ofNat 64 base+offset.signExtend 64*BitVec.ofNat 64 4+BitVec.ofNat 64 k.val).toNat

theorem pointer_exact (base : Nat) (z : Int) (hz : Signed32 z) (nonneg : 0≤z)
    (k : Fin 4) (fits : base+4*z.toNat+k.val<2^64) :
    pointer base (i32 z) k=base+4*z.toNat+k.val := by
  have zn:(z.toNat:Int)=z:=Int.toNat_of_nonneg nonneg
  have word:BitVec.ofInt 64 z=BitVec.ofNat 64 z.toNat := by
    calc BitVec.ofInt 64 z = BitVec.ofInt 64 (z.toNat:Int) := congrArg (BitVec.ofInt 64) zn.symm
         _ = BitVec.ofNat 64 z.toNat := rfl
  rw [pointer,widened z hz,word]
  simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have eq:base+z.toNat*4+k.val=base+4*z.toNat+k.val := by omega
  rw [eq,Nat.mod_eq_of_lt fits]

theorem stored_address (t : TensorView) (words a : Nat) (bound : Stored t words)
    (ha : a<words) (k : Fin 4) :
    t.storageBase≤t.pointer+4*a+k.val ∧
    t.pointer+4*a+k.val<t.storageBase+t.capacity ∧ t.pointer+4*a+k.val<2^64 := by
  obtain ⟨alignment,lo,hi,addr⟩:=bound (by omega)
  have hk:=k.isLt
  omega

/-- Complete actual input pointer at each mask-active source load, bound to
the caller's data_ptr and original storage; no per-coordinate Nat clamp. -/
theorem input_address (c : Call) (d : Domain c) (g r f h w k : Nat)
    (hg : g<c.shape.groups) (hr : r<pSpatial c.shape) (hf : f<pFeatures c.shape)
    (hh : h<c.shape.kh) (hw : w<c.shape.kw) (hk : k<c.shape.padded)
    (active : batch c.shape r<c.shape.n ∧ k<c.shape.cg ∧
      0 ≤ Conv2d.ih c.shape r h ∧ Conv2d.ih c.shape r h<c.shape.h ∧
      0 ≤ Conv2d.iw c.shape r w ∧ Conv2d.iw c.shape r w<c.shape.w) (byte : Fin 4) :
    pointer c.input.pointer (input32 c.shape g r h w k) byte=
      c.input.pointer+4*(inputPointer c.shape g r h w k).toNat+byte.val ∧
    c.input.storageBase≤pointer c.input.pointer (input32 c.shape g r h w k) byte ∧
    pointer c.input.pointer (input32 c.shape g r h w k) byte<c.input.storageBase+c.input.capacity := by
  have safe:=domain_safety c d
  have hb:=safe.inputMemory g r h w k hg active.1 active.2.1 active.2.2.1
    active.2.2.2.1 active.2.2.2.2.1 active.2.2.2.2.2
  have bounds:=stored_address c.input _ _ d.inputStorage hb.2 byte
  have signed:=(safe.pointers g r f h w k hg hr hf hh hw hk).1
  rw [input32_word,pointer_exact _ _ signed hb.1 byte bounds.2.2]
  exact ⟨rfl,bounds.1,bounds.2.1⟩

theorem weight_address (c : Call) (d : Domain c) (g r f h w k : Nat)
    (hg : g<c.shape.groups) (hr : r<pSpatial c.shape) (hf : f<pFeatures c.shape)
    (hh : h<c.shape.kh) (hw : w<c.shape.kw) (hk : k<c.shape.padded)
    (active : f<c.shape.og ∧ k<c.shape.cg) (byte : Fin 4) :
    pointer c.weight.pointer (weight32 c.shape g f h w k) byte=
      c.weight.pointer+4*weightPointer c.shape g f h w k+byte.val ∧
    c.weight.storageBase≤pointer c.weight.pointer (weight32 c.shape g f h w k) byte ∧
    pointer c.weight.pointer (weight32 c.shape g f h w k) byte<c.weight.storageBase+c.weight.capacity := by
  have safe:=domain_safety c d
  have hb:=safe.weightMemory g f h w k hg active.1 active.2 hh hw
  have bounds:=stored_address c.weight _ _ d.weightStorage hb byte
  have signed:=(safe.pointers g r f h w k hg hr hf hh hw hk).2.1
  rw [weight32_word,pointer_exact _ _ signed (Int.natCast_nonneg _) byte (by simpa using bounds.2.2)]
  simpa only [Int.toNat_natCast,eq_self_iff_true,true_and] using And.intro bounds.1 bounds.2.1

/-- Output pointer equality is valid for every mask-active writer, including
all byte lanes and actual nonzero base addresses. -/
theorem output_address (c : Call) (d : Domain c) (out : TensorView) (a : Host.Allocation c out)
    (x : Cell) (hx : c.shape.Active x) (byte : Fin 4) :
    pointer out.pointer (output32 c.shape x) byte=out.pointer+4*outputPointer c.shape x+byte.val ∧
    out.storageBase≤pointer out.pointer (output32 c.shape x) byte ∧
    pointer out.pointer (output32 c.shape x) byte<out.storageBase+out.capacity := by
  have hb:=address_bound c.shape d.valid x hx
  have bounds:=stored_address out _ _ a.bounds hb byte
  have bound:outputPointer c.shape x≤envelope c.shape := by
    have re:=block_cover c.shape.spatial 128 (by decide)
    have fe:=block_cover c.shape.outputs 32 (by decide)
    have outputs:c.shape.og≤c.shape.outputs := by
      unfold Shape.outputs
      have hg:=d.valid.groups
      nlinarith
    have e:=output_bound_padded c.shape d.valid x hx.1
      (lt_of_lt_of_le hx.2.1 re) (lt_of_lt_of_le (lt_of_lt_of_le hx.2.2 outputs) fe)
    have envelopeBound:outputEnvelope c.shape≤envelope c.shape := by unfold envelope;omega
    exact e.trans envelopeBound
  have signed:Signed32 (outputPointer c.shape x) :=
    signed32_of_bound c.shape _ d.signed (by have :=Int.natCast_nonneg (outputPointer c.shape x);omega)
      (by exact_mod_cast bound)
  rw [output32_word,pointer_exact _ _ signed (Int.natCast_nonneg _) byte (by simpa using bounds.2.2)]
  simpa only [Int.toNat_natCast,eq_self_iff_true,true_and] using And.intro bounds.1 bounds.2.1

/-- The source adds four separately formed, sign-extended element offsets
to its pointer. This retains the actual pointer-add staging. -/
def stagedPointer (base : Nat) (a b c d : Int) (byte : Fin 4) : Nat :=
  (BitVec.ofNat 64 base+(i32 a).signExtend 64*BitVec.ofNat 64 4+
    (i32 b).signExtend 64*BitVec.ofNat 64 4+
    (i32 c).signExtend 64*BitVec.ofNat 64 4+
    (i32 d).signExtend 64*BitVec.ofNat 64 4+BitVec.ofNat 64 byte.val).toNat

theorem stagedPointer_eq (base : Nat) (a b c d : Int) (byte : Fin 4)
    (ha : Signed32 a) (hb : Signed32 b) (hc : Signed32 c) (hd : Signed32 d)
    (total : Signed32 (a+b+c+d)) :
    stagedPointer base a b c d byte=pointer base (i32 (a+b+c+d)) byte := by
  simp only [stagedPointer,pointer,widened a ha,widened b hb,widened c hc,widened d hd,widened _ total]
  change (BitVec.ofInt 64 (base:Int)+BitVec.ofInt 64 a*BitVec.ofInt 64 4+
    BitVec.ofInt 64 b*BitVec.ofInt 64 4+BitVec.ofInt 64 c*BitVec.ofInt 64 4+
    BitVec.ofInt 64 d*BitVec.ofInt 64 4+BitVec.ofInt 64 (byte.val:Int)).toNat=
    (BitVec.ofInt 64 (base:Int)+BitVec.ofInt 64 (a+b+c+d)*BitVec.ofInt 64 4+
      BitVec.ofInt 64 (byte.val:Int)).toNat
  simp only [←BitVec.ofInt_add,←BitVec.ofInt_mul]
  congr 2
  ring

def inputNative (base : Nat) (s : Shape) (g r h w c : Nat) (byte : Fin 4) : Nat :=
  stagedPointer base (s.sn*batch s r+s.sc*g*s.cg) (s.sc*c)
    (s.sx*ih s r h) (s.sy*iw s r w) byte

def weightNative (base : Nat) (s : Shape) (g f h w c : Nat) (byte : Fin 4) : Nat :=
  stagedPointer base (s.so*f+s.so*g*s.og) (s.tc*c) (s.tx*h) (s.ty*w) byte

/-- Complete source input-pointer staging reduces to the signed expression
used by `input_address`; every sign extension is justified by checked bounds. -/
theorem input_native_binding (call : Call) (d : Domain call) (base g r h w c : Nat)
    (hg : g<call.shape.groups) (hr : r<pSpatial call.shape) (hh : h<call.shape.kh)
    (hw : w<call.shape.kw) (hc : c<call.shape.padded) (byte : Fin 4) :
    inputNative base call.shape g r h w c byte=
      pointer base (input32 call.shape g r h w c) byte := by
  have formed:=(domain_safety call d).inputFormed g r h w c hg hr hh hw hc
  rw [input32_word]
  apply stagedPointer_eq
  all_goals apply formed
  all_goals simp [inputIntermediates,inputPointer]

/-- The source weight base and three offset additions are likewise bound. -/
theorem weight_native_binding (call : Call) (d : Domain call) (base g f h w c : Nat)
    (hg : g<call.shape.groups) (hf : f<pFeatures call.shape) (hh : h<call.shape.kh)
    (hw : w<call.shape.kw) (hc : c<call.shape.padded) (byte : Fin 4) :
    weightNative base call.shape g f h w c byte=
      pointer base (weight32 call.shape g f h w c) byte := by
  have formed:=(domain_safety call d).weightFormed g f h w c hg hf hh hw hc
  rw [weight32_word]
  change stagedPointer base _ _ _ _ byte=pointer base (i32
    (((call.shape.so:Int)*f+call.shape.so*g*call.shape.og)+call.shape.tc*c+call.shape.tx*h+call.shape.ty*w)) byte
  apply stagedPointer_eq
  all_goals apply formed
  all_goals simp [weightIntermediates,weightPointer]

#axiomsClean stagedPointer_eq
#axiomsClean input_native_binding
#axiomsClean weight_native_binding
#axiomsClean input32_word
#axiomsClean signed_compare
#axiomsClean pointer_exact
#axiomsClean input_address
#axiomsClean weight_address
#axiomsClean output_address
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d.Lowering
