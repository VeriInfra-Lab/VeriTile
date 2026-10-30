import bench.tritonbench_g.rbe_triton_transform.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Index
open VeriTile Triton Binding
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Ordinary supported native launch, separate from the unchanged guards.
The guards alone admit (1,131071,1), whose grid-y is 65536. -/
structure SupportedLaunch (r : Request) : Prop where
  batch : 0<r.size → r.batches≤2^31-1
  program : 0<r.size → grid r≤65535

def pidWord (n : Nat) : BitVec 64 := BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt

theorem signed_exact (n b : Nat) (bound : n<2^(b-1)) (positive : 0<b) :
    (BitVec.ofNat b n).toInt=(n:Int) := by
  have capacity : n<2^b := lt_of_lt_of_le bound (Nat.pow_le_pow_right (by decide) (by omega))
  have nat : (BitVec.ofNat b n).toNat=n := by rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt capacity]
  rw [BitVec.toInt_eq_toNat_of_lt,nat]
  rw [nat]
  have powers : 2^b=2*2^(b-1) := by
    calc
      2^b=2^((b-1)+1) := congrArg (fun n : Nat => 2^n) (by omega)
      _ = _ := by rw [Nat.pow_succ,Nat.mul_comm]
  rw [powers]
  omega

theorem pid_exact (n : Nat) (bound : n<2^31) : pidWord n=BitVec.ofNat 64 n := by
  simp [pidWord,signed_exact n 32 bound (by decide),BitVec.ofInt_natCast]

theorem natural_sign (n : Nat) (bound : n<2^63) : (BitVec.ofNat 64 n).msb=Bool.false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  omega

theorem natural_div (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (BitVec.ofNat 64 a).sdiv (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a/b) := by
  rw [BitVec.sdiv_eq,natural_sign a ha,natural_sign b hb]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.udiv_eq,BitVec.toNat_udiv,BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (by omega : a<2^64),Nat.mod_eq_of_lt (by omega : b<2^64),
    Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.div_le_self a b) (by omega))]

theorem natural_rem (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (BitVec.ofNat 64 a).srem (BitVec.ofNat 64 b)=BitVec.ofNat 64 (a%b) := by
  rw [BitVec.srem_eq,natural_sign a ha,natural_sign b hb]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_umod,BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (by omega : a<2^64),Nat.mod_eq_of_lt (by omega : b<2^64),
    Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.mod_le a b) (by omega))]

def partsWord (r : Request) : BitVec 64 :=
  (BitVec.ofNat 64 r.columns+BitVec.ofNat 64 1024-BitVec.ofNat 64 1).sdiv (BitVec.ofNat 64 1024)

theorem parts_exact (r : Request) (bound : r.columns≤2^62) :
    partsWord r=BitVec.ofNat 64 (parts r) := by
  have sub : BitVec.ofNat 64 (r.columns+1024)-BitVec.ofNat 64 1=BitVec.ofNat 64 (r.columns+1023) := by
    rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by decide) (by omega)]
    congr 1
  rw [partsWord,←BitVec.ofNat_add,sub,natural_div _ _ (by omega) (by decide)]
  rfl

def rowWord (r : Request) (q : Runtime.Program) (i : Lane) : BitVec 64 :=
  (pidWord q.2).sdiv (partsWord r)*BitVec.ofNat 64 2+pidWord i.1.val

def columnWord (r : Request) (q : Runtime.Program) (i : Lane) : BitVec 64 :=
  (pidWord q.2).srem (partsWord r)*BitVec.ofNat 64 1024+
    pidWord i.2.1.val*BitVec.ofNat 64 2

def offsetWord (r : Request) (q : Runtime.Program) (i : Lane) (im : Bool) : BitVec 64 :=
  pidWord q.1*BitVec.ofNat 64 (r.rows*r.columns)+BitVec.ofNat 64 r.columns*rowWord r q i+
    columnWord r q i+BitVec.ofNat 64 (if im then 1 else 0)

def pointer (base : BitVec 64) (r : Request) (q : Runtime.Program) (i : Lane) (im : Bool) :=
  base+offsetWord r q i im*BitVec.ofNat 64 4

theorem size_bound (x : Tensor) (r : Request) (input : Input x r) : r.size≤2^62 := by
  have span := x.view.spanFits
  change x.view.base+4*x.size≤2^64 at span
  rw [input.size] at span
  omega

theorem factors (r : Request) (positive : 0<r.size) :
    0<r.batches ∧ 0<r.rows ∧ 0<r.columns ∧ r.rows≤r.size ∧ r.columns≤r.size := by
  have hb : 0<r.batches := by by_contra h; simp [Request.size,show r.batches=0 by omega] at positive
  have hm : 0<r.rows := by by_contra h; simp [Request.size,show r.rows=0 by omega] at positive
  have hk : 0<r.columns := by by_contra h; simp [Request.size,show r.columns=0 by omega] at positive
  refine ⟨hb,hm,hk,?_,?_⟩
  · exact (Nat.le_mul_of_pos_left _ hb).trans (Nat.le_mul_of_pos_right _ hk)
  · exact Nat.le_mul_of_pos_left _ (Nat.mul_pos hb hm)

theorem scheduled (r : Request) (q : Runtime.Program) (member : q∈Runtime.programs r) (i : Lane) :
    0<r.size ∧ row r (Runtime.location q i Bool.false)<rowParts r*2 ∧
      evenColumn r (Runtime.location q i Bool.false)+1<parts r*1024 := by
  have hp := (Runtime.mem_programs r q).mp member
  have hd : 0<parts r := by by_contra h; simp [grid,show parts r=0 by omega] at hp
  have hm : q.2/parts r<rowParts r :=
    (Nat.div_lt_iff_lt_mul hd).mpr (by simpa [grid] using hp.2)
  have hn := Nat.mod_lt q.2 hd
  have rowsPositive : 0<rowParts r := lt_of_le_of_lt (Nat.zero_le _) hm
  have hr : 0<r.rows := by unfold rowParts at rowsPositive; omega
  have hc : 0<r.columns := by unfold parts at hd; omega
  have positive : 0<r.size := Nat.mul_pos (Nat.mul_pos (by omega) hr) hc
  have hmi := i.1.isLt
  have hni := i.2.1.isLt
  refine ⟨positive,?_,?_⟩ <;>
    dsimp [row,evenColumn,Runtime.location,program,lane] <;> omega

theorem words (x : Tensor) (r : Request) (input : Input x r) (launch : SupportedLaunch r)
    (q : Runtime.Program) (member : q∈Runtime.programs r) (i : Lane) (im : Bool) :
    rowWord r q i=BitVec.ofNat 64 (row r (Runtime.location q i im)) ∧
    columnWord r q i=BitVec.ofNat 64 (evenColumn r (Runtime.location q i im)) ∧
    offsetWord r q i im=BitVec.ofNat 64 (r.offset (batch (Runtime.location q i im))
      (row r (Runtime.location q i im)) (column r (Runtime.location q i im))) := by
  have hp := (Runtime.mem_programs r q).mp member
  have coords := scheduled r q member i
  have dims := factors r coords.1
  have cap := size_bound x r input
  have pb : parts r<2^63 := by unfold parts; omega
  have qb : q.2<2^31 := by have := launch.program coords.1; omega
  have bb : q.1<2^31 := by have := launch.batch coords.1; omega
  have ib : i.1.val<2^31 := by have := i.1.isLt; dsimp [TileIndex] at this; omega
  have jb : i.2.1.val<2^31 := by have := i.2.1.isLt; dsimp [TileIndex] at this; omega
  have hr : rowWord r q i=BitVec.ofNat 64 (row r (Runtime.location q i im)) := by
    simp only [rowWord,parts_exact r (by omega),pid_exact _ qb,pid_exact _ ib,natural_div q.2 (parts r) (by omega) pb,
      ←BitVec.ofNat_mul,←BitVec.ofNat_add,row,Runtime.location,program,lane]
  have hc : columnWord r q i=BitVec.ofNat 64 (evenColumn r (Runtime.location q i im)) := by
    simp only [columnWord,parts_exact r (by omega),pid_exact _ qb,pid_exact _ jb,natural_rem q.2 (parts r) (by omega) pb,
      ←BitVec.ofNat_mul,←BitVec.ofNat_add,evenColumn,Runtime.location,program,lane]
  refine ⟨hr,hc,?_⟩
  simp only [offsetWord,pid_exact _ bb,hr,hc,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  congr 1
  dsimp [Request.offset,column,row,evenColumn,Runtime.location,batch,imaginary,program,lane]
  cases im <;> simp only [↓reduceIte,Bool.false_eq_true] <;> ring

theorem mask (x : Tensor) (r : Request) (input : Input x r) (launch : SupportedLaunch r)
    (q : Runtime.Program) (member : q∈Runtime.programs r) (i : Lane) (im : Bool) :
    ((rowWord r q i).slt (BitVec.ofNat 64 r.rows) &&
      (columnWord r q i+BitVec.ofNat 64 (if im then 1 else 0)).slt (BitVec.ofNat 64 r.columns))=
      decide (active r (Runtime.location q i im)) := by
  have hp := (Runtime.mem_programs r q).mp member
  have coords := scheduled r q member i
  have dims := factors r coords.1
  have cap := size_bound x r input
  have hr : row r (Runtime.location q i im)<2^63 := by
    have h := coords.2.1
    unfold rowParts at h
    change row r (Runtime.location q i im)<(r.rows+1)/2*2 at h
    omega
  have hc : column r (Runtime.location q i im)<2^63 := by
    have h := coords.2.2
    have pad : parts r*1024≤r.columns+1023 := Nat.div_mul_le_self _ _
    change evenColumn r (Runtime.location q i im)+1<parts r*1024 at h
    have top : evenColumn r (Runtime.location q i im)+1<2^63 := by omega
    apply lt_of_le_of_lt ?_ top
    unfold column
    split <;> omega
  rw [(words x r input launch q member i im).1,(words x r input launch q member i im).2.1,
    ←BitVec.ofNat_add]
  change (BitVec.slt (BitVec.ofNat 64 (row r (Runtime.location q i im))) (BitVec.ofNat 64 r.rows) &&
    BitVec.slt (BitVec.ofNat 64 (column r (Runtime.location q i im))) (BitVec.ofNat 64 r.columns))=_
  rw [BitVec.slt_eq_decide,BitVec.slt_eq_decide,
    signed_exact _ 64 hr (by decide),signed_exact _ 64 hc (by decide),
    signed_exact _ 64 (by omega : r.rows<2^63) (by decide),
    signed_exact _ 64 (by omega : r.columns<2^63) (by decide)]
  simp [active,batch,Runtime.location,hp.1]

def frequencyWord (r : Request) (q : Runtime.Program) (i : Lane) : BitVec 64 :=
  (columnWord r q i).srem (BitVec.ofNat 64 128)

theorem frequency (x : Tensor) (r : Request) (input : Input x r) (launch : SupportedLaunch r)
    (q : Runtime.Program) (member : q∈Runtime.programs r) (i : Lane) :
    (frequencyWord r q i).toInt=(evenColumn r (Runtime.location q i Bool.false)%128 : Nat) := by
  have coords := scheduled r q member i
  have dims := factors r coords.1
  have cap := size_bound x r input
  have pad : parts r*1024≤r.columns+1023 := Nat.div_mul_le_self _ _
  have bound : evenColumn r (Runtime.location q i Bool.false)<2^63 := by omega
  rw [frequencyWord,(words x r input launch q member i Bool.false).2.1,
    natural_rem _ _ bound (by decide)]
  apply signed_exact _ 64 _ (by decide)
  have := Nat.mod_lt (evenColumn r (Runtime.location q i Bool.false)) (by decide : 0<128)
  omega


def tokenWord (r : Request) (q : Runtime.Program) (i : Lane) : BitVec 64 :=
  pidWord i.1.val+(BitVec.ofInt 64 r.position+
    (pidWord q.2).sdiv (partsWord r)*BitVec.ofNat 64 2)

theorem tokenWord_eq (r : Request) (q : Runtime.Program) (i : Lane) :
    tokenWord r q i=BitVec.ofInt 64 r.position+rowWord r q i := by
  unfold tokenWord rowWord
  ac_rfl

theorem token (x : Tensor) (r : Request) (input : Input x r) (launch : SupportedLaunch r)
    (q : Runtime.Program) (member : q∈Runtime.programs r) (i : Lane) :
    (tokenWord r q i).toInt=
      r.position+(row r (Runtime.location q i Bool.false) : Int) := by
  have coords := scheduled r q member i
  have lower := input.admitted.lower
  change -(2^63 : Int)≤r.position at lower
  have upper := input.admitted.padded (by rw [←input.size] at coords; exact coords.1)
  have pad : Contract.paddedRows x.actual.metadata=rowParts r*2 := by
    simp [Contract.paddedRows,input.shape,rowParts,Nat.mul_comm]
  rw [pad] at upper
  change r.position+(rowParts r*2 : Nat)-1<(2^63 : Int) at upper
  have bounds : -(2^63 : Int)≤r.position+(row r (Runtime.location q i Bool.false):Int) ∧
      r.position+(row r (Runtime.location q i Bool.false):Int)<2^63 := by omega
  rw [tokenWord_eq,(words x r input launch q member i Bool.false).1,←BitVec.ofInt_natCast,←BitVec.ofInt_add]
  exact BitVec.toInt_ofInt_eq_self (by decide) bounds.1 bounds.2

theorem view_address (t : Tensor) (offset : BitVec 64) (o : Nat)
    (equal : offset=BitVec.ofNat 64 o) (active : o<t.size) :
    (t.actual.dataPtr+offset*BitVec.ofNat 64 4).toNat=t.view.address o ∧
      ∀ byte,byte<4 → t.view.address o+byte<2^64 := by
  have nonempty : 0<t.actual.metadata.numel := by change 0<t.size; omega
  have actual := Torch.FP32.actual_pointer t.actual t.laws t.contiguous nonempty
  change BitVec.ofNat 64 (t.view.address 0)=t.actual.dataPtr at actual
  have span := t.view.spanFits
  change t.view.base+4*t.size≤2^64 at span
  constructor
  · rw [←actual,equal,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
    simp only [Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero]
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  · intro byte hb
    unfold Torch.FP32.DataView.address
    omega

theorem addresses (x y : Tensor) (r : Request) (input : Input x r) (out : Output x y)
    (launch : SupportedLaunch r) (q : Runtime.Program) (member : q∈Runtime.programs r)
    (i : Lane) (im : Bool) (active : active r (Runtime.location q i im)) :
    let o := r.offset (batch (Runtime.location q i im)) (row r (Runtime.location q i im))
      (column r (Runtime.location q i im))
    (pointer x.actual.dataPtr r q i im).toNat=x.view.address o ∧
    (pointer y.actual.dataPtr r q i im).toNat=y.view.address o ∧
    x.view.address o%4=0 ∧ y.view.address o%4=0 ∧
    (∀ byte,byte<4 → x.view.address o+byte<2^64 ∧ y.view.address o+byte<2^64) := by
  have bound := offset_lt r _ _ _ active.1 active.2.1 active.2.2
  have xb := bound
  rw [←input.size] at xb
  have yb := bound
  rw [←input.size,←out.size] at yb
  have eq := (words x r input launch q member i im).2.2
  have xa := view_address x _ _ eq xb
  have ya := view_address y _ _ eq yb
  have alignX := Torch.FP32.dataView_aligned x.actual x.laws x.contiguous input.admitted.aligned
  have alignY := Torch.FP32.dataView_aligned y.actual y.laws y.contiguous out.aligned
  exact ⟨xa.1,ya.1,by simpa [Torch.FP32.DataView.address,Nat.add_mod] using alignX,
    by simpa [Torch.FP32.DataView.address,Nat.add_mod] using alignY,
    fun byte hb => ⟨xa.2 byte hb,ya.2 byte hb⟩⟩

/-- Active output words have one program owner; concurrent distinct programs
cannot race on an output word. Intra-program lane separation is `source_apart`. -/
theorem program_owner (r : Request) (a b : Location) (ha : active r a) (hb : active r b)
    (equal : r.offset (batch a) (row r a) (column r a)=
      r.offset (batch b) (row r b) (column r b)) : batch a=batch b ∧ program a=program b := by
  obtain ⟨batchEq,rowEq,colEq⟩ := offset_injective r ha.2.1 ha.2.2 hb.2.1 hb.2.2 equal
  have ar := (lane a).1.isLt
  have br := (lane b).1.isLt
  have ac := (lane a).2.1.isLt
  have bc := (lane b).2.1.isLt
  have quotient : program a/parts r=program b/parts r := by
    unfold row at rowEq
    dsimp [TileIndex] at ar br
    omega
  have remainder : program a%parts r=program b%parts r := by
    unfold column evenColumn at colEq
    dsimp [TileIndex] at ac bc
    cases ia : imaginary a <;> cases ib : imaginary b <;>
      simp only [ia,ib,Bool.false_eq_true,↓reduceIte] at colEq <;> omega
  have ad := Nat.mod_add_div (program a) (parts r)
  have bd := Nat.mod_add_div (program b) (parts r)
  rw [quotient,remainder] at ad
  exact ⟨batchEq,ad.symm.trans bd⟩


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Index
