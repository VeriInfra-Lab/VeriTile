import bench.tritonbench_g.block_sparse_attn.Proofs.Bounds

namespace VeriTile.Bench.Repairs.BlockSparseAttn.CheckedCSR
open Safety
set_option maxHeartbeats 8000000

/-- Precisely the signed metadata inspected by the wrapper. Reversed rows
remain empty and duplicate nonnegative columns remain separate occurrences. -/
structure Guard (a : Host.Arguments) (rowCount colCount : Nat) : Prop where
  rows32 : rowCount<2^31
  cols32 : colCount<2^31
  rowStride : 0<a.layout.csr.rowStride
  colStride : 0<a.layout.csr.colStride ∧ a.layout.csr.colStride<2^31
  rowEnvelope : (a.layout.shape.layouts-1)*a.layout.csr.rowStride+
    ceil a.layout.shape.length a.layout.shape.blockM+1≤rowCount
  colEnvelope : a.layout.shape.layouts*a.layout.csr.colStride≤colCount
  rowStorage : 4*(a.rows.offset+rowCount)≤a.rows.bytes
  colStorage : 4*(a.cols.offset+colCount)≤a.cols.bytes
  endpoints : ∀ h,h<a.layout.shape.heads → ∀ row,
    row<ceil a.layout.shape.length a.layout.shape.blockM*a.layout.shape.blockM →
    0≤a.layout.csr.first a.layout.shape h row ∧
    a.layout.csr.first a.layout.shape h row≤a.layout.csr.colStride ∧
    0≤a.layout.csr.last a.layout.shape h row ∧
    a.layout.csr.last a.layout.shape h row≤a.layout.csr.colStride
  columns : ∀ h,h<a.layout.shape.heads → ∀ row,
    row<ceil a.layout.shape.length a.layout.shape.blockM*a.layout.shape.blockM →
    ∀ visit,visit<a.layout.csr.count a.layout.shape h row →
    let col:=a.layout.csr.cols (a.layout.csr.slot a.layout.shape h row visit)
    0≤col ∧ col*(a.layout.shape.blockN:Int)<a.layout.shape.length ∧
    (a.layout.evenN=Bool.true →
      (col+1)*(a.layout.shape.blockN:Int)≤a.k.rows ∧
      (col+1)*(a.layout.shape.blockN:Int)≤a.v.rows)

theorem lane_bounds (a : Host.Arguments) (h : Host.Header a) (l : Lane)
    (valid : l.Valid a.layout.shape) :
    (laneWork a.layout.shape l).batch<a.layout.shape.batches ∧
    (laneWork a.layout.shape l).head<a.layout.shape.heads ∧
    (laneWork a.layout.shape l).row<ceil a.layout.shape.length a.layout.shape.blockM*a.layout.shape.blockM ∧
    kvHead a.layout.shape (laneWork a.layout.shape l).head<a.layout.shape.kvHeads := by
  have hh : 0<a.layout.shape.heads := h.shape.2.1
  have head : (laneWork a.layout.shape l).head<a.layout.shape.heads := Nat.mod_lt _ hh
  refine ⟨(Nat.div_lt_iff_lt_mul hh).mpr valid.2.1,head,?_,?_⟩
  · have hp:=Nat.mul_le_mul_right a.layout.shape.blockM
      (show l.rowProgram+1≤ceil a.layout.shape.length a.layout.shape.blockM by have:=valid.1;omega)
    have hr:=valid.2.2.2.1
    change l.rowProgram*a.layout.shape.blockM+l.rowLane<_
    nlinarith
  · exact grouped_head_bound _ h.shape.2.2.1 h.shape.2.2.2.1 _ head

theorem row_endpoints (a : Host.Arguments) (header : Host.Header a)
    (guard : Guard a rowCount colCount) (l : Lane) (valid : l.Valid a.layout.shape) :
    let p:=(((laneWork a.layout.shape l).head%a.layout.shape.layouts)*a.layout.csr.rowStride+l.rowProgram : Nat)
    I32 p ∧ I32 (p+1) ∧ Allocated a.rows p ∧ Allocated a.rows (p+1) := by
  have layout : 0<a.layout.shape.layouts := header.shape.2.2.2.2.2.2.2.2.2
  have hl:=Nat.mod_lt (laneWork a.layout.shape l).head layout
  have hp:=Nat.mul_le_mul_right a.layout.csr.rowStride
    (show (laneWork a.layout.shape l).head%a.layout.shape.layouts≤a.layout.shape.layouts-1 by omega)
  have hr:=valid.1
  have envelope:=guard.rowEnvelope
  have bytes:=guard.rowStorage
  have width:=guard.rows32
  let p:=((laneWork a.layout.shape l).head%a.layout.shape.layouts)*a.layout.csr.rowStride+l.rowProgram
  have bound : p+1<rowCount := by dsimp [p];omega
  have intBound : (p:Int)+1<rowCount := by exact_mod_cast bound
  change I32 (p:Int) ∧ I32 ((p:Int)+1) ∧ Allocated a.rows (p:Int) ∧ Allocated a.rows ((p:Int)+1)
  refine ⟨⟨by omega,by omega⟩,⟨by omega,by omega⟩,⟨by omega,?_⟩,⟨by omega,?_⟩⟩
  · simp only [Int.toNat_natCast];omega
  · rw [show (p:Int)+1=((p+1:Nat):Int) by omega]
    simp only [Int.toNat_natCast];omega

theorem visit_bounds (a : Host.Arguments) (header : Host.Header a)
    (guard : Guard a rowCount colCount) (l : Lane) (valid : l.Valid a.layout.shape)
    (visit : Nat) (hv : visit<a.layout.csr.count a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row)
    (n : Nat) (hn : n<a.layout.shape.blockN) :
    let w:=laneWork a.layout.shape l
    let base:=((w.head%a.layout.shape.layouts)*a.layout.csr.colStride : Nat)
    let first:=a.layout.csr.first a.layout.shape w.head w.row
    let last:=a.layout.csr.last a.layout.shape w.head w.row
    let slot:=a.layout.csr.slot a.layout.shape w.head w.row visit
    let col:=a.layout.csr.cols slot
    let key:=a.layout.csr.key a.layout.shape w.head w.row visit n
    I32 base ∧ I32 first ∧ I32 last ∧ I32 (first+visit) ∧ I32 (first+visit+1) ∧
    I32 slot ∧ I32 col ∧ I32 (col*(a.layout.shape.blockN:Int)) ∧ I32 key ∧
    Allocated a.cols slot ∧ 0≤key ∧ key<a.layout.shape.length+a.layout.shape.blockN ∧
    (a.layout.evenN=Bool.true → key<a.k.rows ∧ key<a.v.rows) := by
  obtain ⟨hb,hh,hr,hkv⟩:=lane_bounds a header l valid
  obtain ⟨first0,firstMax,last0,lastMax⟩:=guard.endpoints _ hh _ hr
  obtain ⟨col0,colMax,even⟩:=guard.columns _ hh _ hr visit hv
  have live:=(csr_visit a.layout.shape a.layout.csr _ _ visit).mp hv
  have layouts : 0<a.layout.shape.layouts := header.shape.2.2.2.2.2.2.2.2.2
  have hm:=Nat.mod_lt (laneWork a.layout.shape l).head layouts
  have hbmax:=Nat.mul_le_mul_right a.layout.csr.colStride
    (show (laneWork a.layout.shape l).head%a.layout.shape.layouts+1≤a.layout.shape.layouts by omega)
  rw [Nat.add_mul,Nat.one_mul] at hbmax
  have cstride:=guard.colStride
  have csize:=guard.colEnvelope
  have c32:=guard.cols32
  have bytes:=guard.colStorage
  have bn : 0<a.layout.shape.blockN := header.shape.2.2.2.2.2.2.1
  have length32 : a.layout.shape.length+a.layout.shape.blockN<2^31 := by
    have formed:=header.kguard.formed
    have shape:=header.kshape.2.2.1
    have positive:=header.kguard.positive.2.2.1
    have stride:=(header.kguard.metadata a.k.strides.row (by simp)).1
    unfold Host.Tensor.last at formed
    have hmul:=Nat.le_mul_of_pos_right (a.k.rows-1) stride
    have extra:=Nat.le_mul_of_pos_right (a.layout.shape.blockM+a.layout.shape.blockN) stride
    have bm : 0<a.layout.shape.blockM := header.shape.2.2.2.2.2.1
    omega
  dsimp only
  have slot0 : 0≤a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit := by
    unfold CSR.slot;omega
  have slotMax : a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit<colCount := by
    unfold CSR.slot
    omega
  have start0 : 0≤a.layout.csr.cols (a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row visit)*(a.layout.shape.blockN:Int) := mul_nonneg col0 (by omega)
  have colBound : a.layout.csr.cols (a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row visit)<2^31 := by
    nlinarith
  have key0 : 0≤a.layout.csr.key a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit n := by
    unfold CSR.key;omega
  have keyMax : a.layout.csr.key a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit n<
      a.layout.shape.length+a.layout.shape.blockN := by unfold CSR.key;omega
  refine ⟨⟨by omega,by omega⟩,⟨by omega,by omega⟩,⟨by omega,by omega⟩,
    ⟨by omega,by omega⟩,⟨by omega,by omega⟩,⟨by omega,by omega⟩,
    ⟨by omega,colBound⟩,⟨by omega,by omega⟩,⟨by omega,by omega⟩,
    ⟨slot0,?_⟩,key0,keyMax,?_⟩
  · have hs : (a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head
        (laneWork a.layout.shape l).row visit).toNat<colCount := by omega
    omega
  · intro he
    obtain ⟨ek,ev⟩:=even he
    unfold CSR.key
    constructor <;> nlinarith

theorem signed_address (t : Strides) (b h d : Nat) (key : Int) (nonnegative : 0≤key) :
    signedAddress t b h key d=(address t b h key.toNat d : Int) := by
  simp only [signedAddress,address,Nat.cast_add,Nat.cast_mul,Int.toNat_of_nonneg nonnegative]

theorem tensor_signed (t : Host.Tensor) (s : Shape) (guard : Host.TensorGuard t s)
    (length : s.length≤t.rows) (bm : 0<s.blockM) (b h d : Nat) (key : Int)
    (hb : b<t.batches) (hh : h<t.heads) (hd : d<t.dim)
    (key0 : 0≤key) (keyMax : key<s.length+s.blockN) :
    0≤signedAddress t.strides b h key d ∧ signedAddress t.strides b h key d<2^31 := by
  have rows:=guard.positive.2.2.1
  have hr : key.toNat≤t.rows-1+(s.blockM+s.blockN) := by omega
  have bound:=Host.tensor_formed t s guard b h key.toNat d hb hh hr hd
  rw [signed_address t.strides b h d key key0]
  exact ⟨by omega,by exact_mod_cast bound⟩

theorem key_value_bounds (a : Host.Arguments) (header : Host.Header a)
    (guard : Guard a rowCount colCount) (l : Lane) (valid : l.Valid a.layout.shape)
    (visit : Nat) (hv : visit<a.layout.csr.count a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row)
    (db d n : Nat) (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD) (hn : n<a.layout.shape.blockN) :
    let w:=laneWork a.layout.shape l
    let key:=a.layout.csr.key a.layout.shape w.head w.row visit n
    (0≤signedAddress a.layout.k w.batch (kvHead a.layout.shape w.head) key (db*a.layout.shape.blockD+d) ∧
     signedAddress a.layout.k w.batch (kvHead a.layout.shape w.head) key (db*a.layout.shape.blockD+d)<2^31) ∧
    (0≤signedAddress a.layout.v w.batch (kvHead a.layout.shape w.head) key (l.channelProgram*a.layout.shape.blockD+d) ∧
     signedAddress a.layout.v w.batch (kvHead a.layout.shape w.head) key (l.channelProgram*a.layout.shape.blockD+d)<2^31) ∧
    ((a.layout.evenN=Bool.true ∨ key<(a.layout.shape.length:Int)) →
      Allocated a.k.storage (signedAddress a.layout.k w.batch (kvHead a.layout.shape w.head) key (db*a.layout.shape.blockD+d)) ∧
      Allocated a.v.storage (signedAddress a.layout.v w.batch (kvHead a.layout.shape w.head) key (l.channelProgram*a.layout.shape.blockD+d))) := by
  obtain ⟨hb,hh,hr,hkv⟩:=lane_bounds a header l valid
  obtain ⟨_,_,_,_,_,_,_,_,_,_,key0,keyMax,padded⟩:=visit_bounds a header guard l valid visit hv n hn
  have kd:=channel_bound a.layout.shape db d hdb hd
  have vd:=channel_bound a.layout.shape l.channelProgram d valid.2.2.1 hd
  have bm : 0<a.layout.shape.blockM := header.shape.2.2.2.2.2.1
  dsimp only
  rw [header.strides.2.1,header.strides.2.2]
  refine ⟨tensor_signed a.k a.layout.shape header.kguard header.kshape.2.2.1 bm _ _ _ _
    (header.kshape.1.symm ▸ hb) (header.kshape.2.1.symm ▸ hkv) (header.kshape.2.2.2.symm ▸ kd) key0 keyMax,
    tensor_signed a.v a.layout.shape header.vguard header.vshape.2.2.1 bm _ _ _ _
    (header.vshape.1.symm ▸ hb) (header.vshape.2.1.symm ▸ hkv) (header.vshape.2.2.2.symm ▸ vd) key0 keyMax,?_⟩
  intro enabled
  have rows : a.layout.csr.key a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit n<a.k.rows ∧
      a.layout.csr.key a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row visit n<a.v.rows := by
    rcases enabled with even|masked
    · exact padded even
    · have hk:=header.kshape.2.2.1;have hv:=header.vshape.2.2.1;constructor <;> omega
  rw [signed_address _ _ _ _ _ key0,signed_address _ _ _ _ _ key0]
  exact ⟨Host.tensor_access a.k a.layout.shape header.kguard _ _ _ _
    (header.kshape.1.symm ▸ hb) (header.kshape.2.1.symm ▸ hkv) (by omega) (header.kshape.2.2.2.symm ▸ kd),
    Host.tensor_access a.v a.layout.shape header.vguard _ _ _ _
    (header.vshape.1.symm ▸ hb) (header.vshape.2.1.symm ▸ hkv) (by omega) (header.vshape.2.2.2.symm ▸ vd)⟩

theorem formed (a : Host.Arguments) (header : Host.Header a)
    (guard : Guard a rowCount colCount) (l : Lane) (valid : l.Valid a.layout.shape)
    (visit : Nat) (hv : visit<a.layout.csr.count a.layout.shape (laneWork a.layout.shape l).head (laneWork a.layout.shape l).row)
    (db d n : Nat) (hdb : db<a.layout.shape.blocks) (hd : d<a.layout.shape.blockD) (hn : n<a.layout.shape.blockN)
    (z : Int) (hz : z∈dataNodes a.layout l visit db d n) : I32 z := by
  obtain ⟨base,first,last,iter,next,slot,col,start,key,allocated,key0,keyMax,padded⟩:=
    visit_bounds a header guard l valid visit hv n hn
  obtain ⟨kb,vb,access⟩:=key_value_bounds a header guard l valid visit hv db d n hdb hd hn
  obtain ⟨hb,hh,hr,hkv⟩:=lane_bounds a header l valid
  obtain ⟨first0,_,last0,_⟩:=guard.endpoints _ hh _ hr
  obtain ⟨col0,_,_⟩:=guard.columns _ hh _ hr visit hv
  have sk : 0≤a.layout.csr.cols (a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row visit)*(a.layout.shape.blockN:Int)*(a.layout.k.row:Int) := by positivity
  have sv : 0≤a.layout.csr.cols (a.layout.csr.slot a.layout.shape (laneWork a.layout.shape l).head
      (laneWork a.layout.shape l).row visit)*(a.layout.shape.blockN:Int)*(a.layout.v.row:Int) := by positivity
  have k1 : (0:Int)≤(n:Int)*a.layout.k.row := by positivity
  have k2 : (0:Int)≤(d:Int)*a.layout.k.dim := by positivity
  have k3 : (0:Int)≤(db:Int)*a.layout.shape.blockD*a.layout.k.dim := by positivity
  have k4 : (0:Int)≤((laneWork a.layout.shape l).batch:Int)*a.layout.k.batch := by positivity
  have k5 : (0:Int)≤(kvHead a.layout.shape (laneWork a.layout.shape l).head:Int)*a.layout.k.head := by positivity
  have v1 : (0:Int)≤(n:Int)*a.layout.v.row := by positivity
  have v2 : (0:Int)≤(d:Int)*a.layout.v.dim := by positivity
  have v3 : (0:Int)≤(l.channelProgram:Int)*a.layout.shape.blockD*a.layout.v.dim := by positivity
  have v4 : (0:Int)≤((laneWork a.layout.shape l).batch:Int)*a.layout.v.batch := by positivity
  have v5 : (0:Int)≤(kvHead a.layout.shape (laneWork a.layout.shape l).head:Int)*a.layout.v.head := by positivity
  dsimp only [dataNodes] at hz
  simp only [List.mem_cons,List.mem_nil_iff,or_false,Nat.cast_add,Nat.cast_mul,add_assoc] at hz
  simp only [I32,CSR.slot,CSR.key,Nat.cast_add,Nat.cast_mul,add_assoc] at base first last iter next slot col start key col0 sk sv
  simp only [signedAddress,CSR.key,CSR.slot,Nat.cast_add,Nat.cast_mul,add_assoc] at kb vb
  rcases hz with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [I32,signedAddress,Nat.cast_add,Nat.cast_mul,add_assoc]
  all_goals try assumption
  all_goals constructor <;> nlinarith only [kb.2,vb.2,sk,sv,k1,k2,k3,k4,k5,v1,v2,v3,v4,v5]

/-- The old safety contract is now a consequence of inspected CSR metadata,
header bounds and complete current storage extents, not a caller assumption. -/
theorem data_contract (a : Host.Arguments) (header : Host.Header a)
    (guard : Guard a rowCount colCount) : DataContract a.layout a.rows a.cols a.k.storage a.v.storage := by
  constructor
  · exact row_endpoints a header guard
  · intro l hl visit hv db hdb d hd n hn z hz
    exact formed a header guard l hl visit hv db d n hdb hd hn z hz
  · intro l hl visit hv
    have bn : 0<a.layout.shape.blockN := header.shape.2.2.2.2.2.2.1
    exact (visit_bounds a header guard l hl visit hv 0 bn).2.2.2.2.2.2.2.2.2.1
  · intro l hl visit hv db hdb d hd n hn
    exact (key_value_bounds a header guard l hl visit hv db d n hdb hd hn).2.2

theorem contract (a : Host.Arguments) (header : Host.Header a) (guard : Guard a rowCount colCount) :
    Host.Contract a := ⟨header,data_contract a header guard⟩

#axiomsClean row_endpoints
#axiomsClean visit_bounds
#axiomsClean key_value_bounds
#axiomsClean formed
#axiomsClean data_contract
#axiomsClean contract
#specNonCircular Guard avoiding [Host.invoke,execute,rowValue,Spec.output,Safety.DataContract]
end VeriTile.Bench.Repairs.BlockSparseAttn.CheckedCSR
