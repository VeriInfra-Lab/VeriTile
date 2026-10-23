import bench.repairs.tritonbench_g.chunk_delta_fwd.Contract

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.NativePointers
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
set_option maxHeartbeats 8000000

def word32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
def word64 (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

theorem exact32 {n : Nat} (h : n < 2^31) : (word32 n).toInt=(n : Int) := by
  have hn : (word32 n).toNat=n:=Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]
theorem exact64 {n : Nat} (h : n < 2^63) : (word64 n).toInt=(n : Int) := by
  have hn : (word64 n).toNat=n:=Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn];omega),hn]

inductive Site where | key | weight | value | corrected | history | initial | final
  deriving DecidableEq

def region : Site → RegionName
  | .key=>"key" | .weight=>"weight" | .value=>"value" | .corrected=>"corrected"
  | .history=>"history" | .initial=>"initial" | .final=>"final"

/-- Coordinates include masked lanes. The bounds allow one-past endpoints. -/
structure Coordinates (c : Host.Dimensions) where
  (head batch headInBatch chunk key time value : Nat)
  headBound : head ≤ c.B*c.H
  batchBound : batch ≤ c.B
  headInBatchBound : headInBatch ≤ c.H
  chunkBound : chunk ≤ c.NT
  keyBound : key ≤ c.BK
  timeBound : time ≤ timeExtent c
  valueBound : value ≤ valueExtent c

def base (c : Host.Dimensions) (x : Coordinates c) : Site → Nat
  | .key=>x.head*(c.T*c.K)
  | .weight=>x.batch*c.wBatch+x.headInBatch*c.wHead
  | .value | .corrected=>x.head*(c.T*c.V)
  | .history=>x.head*(c.NT*c.K*c.V)+x.chunk*c.K*c.V
  | .initial | .final=>x.head*c.K*c.V

def baseWord (c : Host.Dimensions) (x : Coordinates c) : Site → BitVec 32
  | .key=>word32 x.head*word32 (c.T*c.K)
  | .weight=>word32 x.batch*word32 c.wBatch+word32 x.headInBatch*word32 c.wHead
  | .value | .corrected=>word32 x.head*word32 (c.T*c.V)
  | .history=>word32 x.head*word32 (c.NT*c.K*c.V)+word32 x.chunk*word32 c.K*word32 c.V
  | .initial | .final=>word32 x.head*word32 c.K*word32 c.V

def blockOffset (c : Host.Dimensions) (x : Coordinates c) : Site → Nat
  | .key=>x.key+x.time*c.K
  | .weight=>x.time*c.wTime+x.key*c.wDim
  | .value | .corrected=>x.time*c.V+x.value
  | .history | .initial | .final=>x.key*c.V+x.value

def blockWord (c : Host.Dimensions) (x : Coordinates c) : Site → BitVec 64
  | .key=>word64 x.key+word64 x.time*word64 c.K
  | .weight=>word64 x.time*word64 c.wTime+word64 x.key*word64 c.wDim
  | .value | .corrected=>word64 x.time*word64 c.V+word64 x.value
  | .history | .initial | .final=>word64 x.key*word64 c.V+word64 x.value

def offset (c : Host.Dimensions) (x : Coordinates c) (site : Site) := base c x site+blockOffset c x site

theorem formed (c : Host.Dimensions) (x : Coordinates c) (site : Site) : offset c x site < formedExtent c := by
  have hk : x.head*(c.T*c.K)+x.key+x.time*c.K ≤ keyExtent c := by
    unfold keyExtent;simp only [←Nat.mul_assoc];gcongr  <;> first | exact x.headBound | exact x.keyBound | exact x.timeBound
  have hw : x.batch*c.wBatch+x.headInBatch*c.wHead+x.time*c.wTime+x.key*c.wDim ≤ weightExtent c := by
    unfold weightExtent;gcongr  <;> first | exact x.batchBound | exact x.headInBatchBound | exact x.timeBound | exact x.keyBound
  have hv : x.head*(c.T*c.V)+x.time*c.V+x.value ≤ valueOffsetExtent c := by
    unfold valueOffsetExtent;simp only [←Nat.mul_assoc];gcongr  <;> first | exact x.headBound | exact x.timeBound | exact x.valueBound
  have hh : x.head*(c.NT*c.K*c.V)+x.chunk*c.K*c.V+x.key*c.V+x.value ≤ historyExtent c := by
    unfold historyExtent;simp only [←Nat.mul_assoc];gcongr  <;> first | exact x.headBound | exact x.chunkBound | exact x.keyBound | exact x.valueBound
  have hs : x.head*c.K*c.V+x.key*c.V+x.value ≤ stateExtent c := by
    unfold stateExtent;gcongr  <;> first | exact x.headBound | exact x.keyBound | exact x.valueBound
  cases site  <;> simp only [offset,base,blockOffset]  <;> unfold formedExtent  <;> omega

/-- Actual source base operations are signed i32. Descriptor offsets are i64.
The checked total bounds both and every nonnegative partial sum/product. -/
theorem native (r : Request) (checked : r.check=Bool.true) (x : Coordinates r.dims)
    (site : Site) (tensor : Tensor) (pointer : Pointer tensor (formedExtent r.dims)) :
    (baseWord r.dims x site).toInt=(base r.dims x site : Int) ∧
    (blockWord r.dims x site).toInt=(blockOffset r.dims x site : Int) ∧
    (word64 tensor.byteBase+((baseWord r.dims x site).signExtend 64+blockWord r.dims x site)*word64 4).toNat=
      tensor.byteBase+4*offset r.dims x site ∧
    (tensor.byteBase+4*offset r.dims x site)%4=0 ∧
    tensor.byteBase+4*offset r.dims x site+4 < 2^63 := by
  have ho:=formed r.dims x site
  have limit:=r.formed checked
  have b32 : baseWord r.dims x site=word32 (base r.dims x site) := by
    cases site  <;> simp [baseWord,base,word32,←BitVec.ofNat_add,←BitVec.ofNat_mul]
  have b64 : blockWord r.dims x site=word64 (blockOffset r.dims x site) := by
    cases site  <;> simp [blockWord,blockOffset,word64,←BitVec.ofNat_add,←BitVec.ofNat_mul]
  have hb : base r.dims x site < 2^31:=by unfold offset at ho;omega
  have hd : blockOffset r.dims x site < 2^63:=by unfold offset at ho;omega
  have widened : (baseWord r.dims x site).signExtend 64=word64 (base r.dims x site) := by
    apply BitVec.eq_of_toInt_eq
    rw [BitVec.toInt_signExtend_of_le (by decide),b32,exact32 hb,exact64 (by omega : base r.dims x site < 2^63)]
  rw [widened,b32,b64]
  refine ⟨exact32 hb,exact64 hd,?_,?_,?_⟩
  · simp only [word64,←BitVec.ofNat_add,←BitVec.ofNat_mul,BitVec.toNat_ofNat]
    have hmod : tensor.byteBase+(base r.dims x site+blockOffset r.dims x site)*4 < 2^64:=by
      have:=pointer.2;unfold offset at ho;omega
    rw [Nat.mod_eq_of_lt hmod]
    simp [offset,Nat.mul_comm]
  · have:=pointer.1;omega
  · have:=pointer.2;omega

theorem time_formed (c : Host.Dimensions) (outer inner lane : Nat)
    (ho : outer < c.NT) (hi : inner  <  innerCount c.parameters) (hl : lane < c.BC) :
    time c.parameters outer inner lane ≤ timeExtent c := by
  have ha:=Nat.mul_le_mul_right c.BT (Nat.le_of_lt ho)
  have hb:=Nat.mul_le_mul_right c.BC (Nat.succ_le_of_lt hi)
  change (inner+1)*c.BC ≤ ceilDiv c.BT c.BC*c.BC at hb
  change outer*c.BT+inner*c.BC+lane ≤ c.NT*c.BT+ceilDiv c.BT c.BC*c.BC
  simp only [Nat.add_mul,Nat.one_mul] at hb
  omega

theorem value_formed (c : Host.Dimensions) (pid : PID) (valid : Geometry.ValidPID c.parameters pid)
    (lane : Nat) (hl : lane < c.BV) : column c.parameters pid lane ≤ valueExtent c := by
  have hb:=Nat.mul_le_mul_right c.BV (Nat.succ_le_of_lt valid.1)
  change (pid.1+1)*c.BV ≤ ceilDiv c.V c.BV*c.BV at hb
  change pid.1*c.BV+lane ≤ ceilDiv c.V c.BV*c.BV
  simp only [Nat.add_mul,Nat.one_mul] at hb
  omega


/-- State/history sites also exist when T=0; their coordinates do not demand
an inner-loop iteration. -/
def stateCoordinates (r : Request) (pid : PID) (valid : Geometry.ValidPID r.params pid)
    (chunk : Nat) (hc : chunk ≤ r.dims.NT) (key : Fin r.dims.BK) (lane : Fin r.dims.BV) : Coordinates r.dims := by
  have hp : 0 < r.H := by
    have h:=valid.2
    change pid.2 < r.B*r.H at h
    by_contra hn
    have hz : r.H=0:=by omega
    simp [hz] at h
  refine ⟨pid.2,pid.2/r.H,pid.2%r.H,chunk,key.val,0,column r.params pid lane.val,
    valid.2.le,?_,?_,hc,key.isLt.le,Nat.zero_le _,?_⟩
  · exact (Nat.div_lt_iff_lt_mul hp).mpr valid.2 |>.le
  · exact (Nat.mod_lt _ hp).le
  · exact value_formed r.dims pid valid lane.val lane.isLt

def innerCoordinates (r : Request) (pid : PID) (valid : Geometry.ValidPID r.params pid)
    (chunk : Nat) (hc : chunk < r.dims.NT) (block : Nat) (hb : block  <  innerCount r.params)
    (key : Fin r.dims.BK) (lane : Fin r.dims.BV) (timeLane : Fin r.dims.BC) : Coordinates r.dims :=
  {stateCoordinates r pid valid chunk hc.le key lane with
    time:=time r.params chunk block timeLane.val
    timeBound:=time_formed r.dims chunk block timeLane.val hc hb timeLane.isLt}

theorem time_extent_bound (r : Request) (checked : r.check=Bool.true) : timeExtent r.dims < 2^31 := by
  have positive : 1 ≤ r.dims.K:=(r.check_iff.mp checked).2.2.1
  have product : timeExtent r.dims ≤ timeExtent r.dims*r.dims.K:=Nat.le_mul_of_pos_right _ positive
  have limit:=r.formed checked
  unfold formedExtent keyExtent at limit
  omega

theorem time_native (r : Request) (checked : r.check=Bool.true) (chunk block lane : Nat)
    (hc : chunk < r.dims.NT) (hb : block  <  innerCount r.params) (hl : lane < r.dims.BC) :
    (word32 chunk*word32 r.BT+word32 block*word32 r.dims.BC+word32 lane).toInt=
      (time r.params chunk block lane : Int) := by
  have bound:=time_formed r.dims chunk block lane hc hb hl
  change time r.params chunk block lane  ≤  timeExtent r.dims at bound
  have limit:=time_extent_bound r checked
  have eq : word32 chunk*word32 r.BT+word32 block*word32 r.dims.BC+word32 lane=
      word32 (time r.params chunk block lane):=by
    simp [word32,time,Request.params,Host.Dimensions.parameters,Request.dims,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  rw [eq]
  exact exact32 (by omega)

theorem loop_bounds (r : Request) (checked : r.check=Bool.true) : r.dims.NT < 2^31 ∧ innerCount r.params < 2^31 := by
  have all:=(r.check_iff.mp checked).2.2.2.2.2.2.2.1
  have bt : r.BT < 2^31:=all _ (by simp [Request.scalarExtents])
  have nt : r.dims.NT < 2^31:=all _ (by simp [Request.scalarExtents])
  have bc : 1 ≤ r.dims.BC:=(r.shape checked).bc_pos
  have small : innerCount r.params ≤ r.BT := by
    have div:=Nat.div_mul_le_self (r.BT+r.dims.BC-1) r.dims.BC
    change innerCount r.params*r.dims.BC ≤ r.BT+r.dims.BC-1 at div
    by_contra hn
    have hmul:=Nat.mul_le_mul_right r.dims.BC (show r.BT+1 ≤ innerCount r.params by omega)
    have hmul2:=Nat.mul_le_mul_left r.BT bc
    simp only [Nat.add_mul,Nat.one_mul,Nat.mul_one] at hmul hmul2
    omega
  exact ⟨nt,lt_of_le_of_lt small bt⟩

theorem range (bound : Nat) (hb : bound < 2^31) :
    (word32 0).toInt=0 ∧ (word32 bound).toInt=(bound : Int) ∧ (word32 1).toInt=1 ∧
    (∀ i,i ≤ bound → ((word32 i).toInt < (word32 bound).toInt ↔ i < bound)) ∧
    (∀ i,i < bound → (word32 i+word32 1).toInt=(i+1 : Nat)) := by
  refine ⟨exact32 (by norm_num),exact32 hb,exact32 (by norm_num),?_,?_⟩
  · intro i hi;rw [exact32 (by omega : i < 2^31),exact32 hb];omega
  · intro i hi
    rw [show word32 i+word32 1=word32 (i+1) by simp [word32,←BitVec.ofNat_add]]
    exact exact32 (by omega)

#axiomsClean native
#axiomsClean range
#axiomsClean stateCoordinates
#axiomsClean innerCoordinates
#axiomsClean time_native
#axiomsClean loop_bounds
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked.NativePointers
