import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeScalar

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeAddress
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeWord
set_option maxHeartbeats 4000000

/-- Actual allocation base addresses and finite FP32 windows. The address-end
bound is successful pointer/allocation metadata, not a Python shape filter. -/
structure Allocation (a : FPFlat.Placement) where
  byteBase : RegionName → Nat
  addressEnd : ∀ r, byteBase (a.region r)+(a.base r+a.extent r)*4 ≤ 2^64

def base (a : FPFlat.Placement) (alloc : Allocation a) (r : RegionName) : Word :=
  enc (alloc.byteBase (a.region r)+a.base r*4)
def pointer (a : FPFlat.Placement) (alloc : Allocation a) (r : RegionName) (off : Nat) :=
  base a alloc r+enc off*enc 4
def decode (a : FPFlat.Placement) (alloc : Allocation a) (r : RegionName) (ptr : Word) :=
  (ptr.toNat-alloc.byteBase (a.region r))/4

theorem pointer_word (a : FPFlat.Placement) (alloc : Allocation a) (r : RegionName) (off : Nat) :
    pointer a alloc r off = enc (alloc.byteBase (a.region r)+(a.base r+off)*4) := by
  simp only [pointer,base,enc_mul,enc_add]
  congr 1
  ring

theorem decode_exact (a : FPFlat.Placement) (alloc : Allocation a) (r : RegionName) (off : Nat)
    (bound : off < a.extent r) :
    decode a alloc r (pointer a alloc r off) = FPFlat.address a r off := by
  unfold decode
  rw [pointer_word,enc_nat _ (by have hh := alloc.addressEnd r; omega)]
  rw [Nat.add_sub_cancel_left,Nat.mul_div_cancel _ (by decide)]
  rfl

/-- Native pointer recurrence, including the last non-observed increment. -/
def advance (initial increment : Word) : Nat → Word
  | 0 => initial
  | n+1 => advance initial increment n+increment

theorem advance_word (initial : Word) (increment n : Nat) :
    advance initial (enc increment) n = initial+enc (n*increment) := by
  induction n with
  | zero => simp [advance,enc]
  | succ n ih =>
    rw [advance,ih,BitVec.add_assoc,enc_add]
    congr 1
    congr 1
    ring

def dOffset (h : HostCall) (c : Tuning) (q : HostFP.Pid) (step : Nat) (i : TileIndex [c.BM,c.BCS]) :=
  let p := HostContract.parameters h
  q.2.1*p.dBatch+FPLogical.chunk h q*p.dChunk+FPLogical.head h q*p.dHead+
    (FPLogical.row h c q i.1*p.dColumn+(step*c.BCS+i.2.1.val)*p.dRow)
def aOffset (h : HostCall) (c : Tuning) (q : HostFP.Pid) (step : Nat) (i : TileIndex [c.BCS,c.BN]) :=
  let p := HostContract.parameters h
  q.2.1*p.aBatch+FPLogical.chunk h q*p.chunk*p.aRow+FPLogical.head h q*p.aHead+
    ((step*c.BCS+i.1.val)*p.aRow+FPLogical.column h c q i.2.1*p.aColumn)
def tensorOffset (h : HostCall) (c : Tuning) (q : HostFP.Pid)
    (sb ss sg sk : Nat) (i : TileIndex [c.BM,c.BN]) :=
  q.2.1*sb+FPLogical.chunk h q*h.chunk*ss+FPLogical.head h q*sg+
    (FPLogical.row h c q i.1*ss+FPLogical.column h c q i.2.1*sk)

def dout (h : HostCall) (c : Tuning) (a : FPFlat.Placement) (alloc : Allocation a)
    (q : HostFP.Pid) (i : TileIndex [c.BM,c.BCS]) (step : Nat) : Word :=
  let p := HostContract.parameters h
  let s := NativeScalar.setup h c q
  let ptr := base a alloc h.Dout+(s.batch*enc p.dBatch+s.chunk*enc p.dChunk+s.head*enc p.dHead)*enc 4
  let ptr := ptr+(NativeScalar.row s c.BM i.1*enc p.dColumn+enc i.2.1.val*enc p.dRow)*enc 4
  advance ptr (enc c.BCS*enc p.dRow*enc 4) step

def tensorA (h : HostCall) (c : Tuning) (a : FPFlat.Placement) (alloc : Allocation a)
    (q : HostFP.Pid) (i : TileIndex [c.BCS,c.BN]) (step : Nat) : Word :=
  let p := HostContract.parameters h
  let s := NativeScalar.setup h c q
  let ptr := base a alloc h.A+(s.batch*enc p.aBatch+s.chunk*enc h.chunk*enc p.aRow+s.head*enc p.aHead)*enc 4
  let ptr := ptr+(enc i.1.val*enc p.aRow+NativeScalar.column s c.BN i.2.1*enc p.aColumn)*enc 4
  advance ptr (enc c.BCS*enc p.aRow*enc 4) step

def tensor (h : HostCall) (c : Tuning) (a : FPFlat.Placement) (alloc : Allocation a)
    (q : HostFP.Pid) (r : RegionName) (sb ss sg sk : Nat) (i : TileIndex [c.BM,c.BN]) : Word :=
  let s := NativeScalar.setup h c q
  (base a alloc r+(s.batch*enc sb+s.chunk*enc h.chunk*enc ss+s.head*enc sg)*enc 4)+
    (NativeScalar.row s c.BM i.1*enc ss+NativeScalar.column s c.BN i.2.1*enc sk)*enc 4

/-- Exact modulo correspondence for every pointer lane and every pointer
advance, including inactive lanes. Active dereference exactness is separate. -/
theorem dout_word (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (a : FPFlat.Placement) (alloc : Allocation a) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (i : TileIndex [(tuning choice).BM,(tuning choice).BCS]) (step : Nat) :
    dout h (tuning choice) a alloc q i step = pointer a alloc h.Dout (dOffset h (tuning choice) q step i) := by
  dsimp only [dout]
  rw [NativeScalar.setup_exact h choice metadata q hq,NativeScalar.row_exact]
  simp only [NativeScalar.ideal,enc_mul,advance_word,base,pointer,enc_add,dOffset]
  congr 1
  ring

theorem a_word (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (a : FPFlat.Placement) (alloc : Allocation a) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (i : TileIndex [(tuning choice).BCS,(tuning choice).BN]) (step : Nat) :
    tensorA h (tuning choice) a alloc q i step = pointer a alloc h.A (aOffset h (tuning choice) q step i) := by
  dsimp only [tensorA]
  rw [NativeScalar.setup_exact h choice metadata q hq,NativeScalar.column_exact]
  simp only [NativeScalar.ideal,enc_mul,advance_word,base,pointer,enc_add,aOffset,HostContract.parameters]
  congr 1
  ring

theorem tensor_word (h : HostCall) (choice : Fin 9) (metadata : NativeControl.Metadata h (tuning choice))
    (a : FPFlat.Placement) (alloc : Allocation a) (q : HostFP.Pid) (hq : q ∈ HostFP.grid h (tuning choice))
    (r : RegionName) (sb ss sg sk : Nat) (i : TileIndex [(tuning choice).BM,(tuning choice).BN]) :
    tensor h (tuning choice) a alloc q r sb ss sg sk i =
      pointer a alloc r (tensorOffset h (tuning choice) q sb ss sg sk i) := by
  dsimp only [tensor]
  rw [NativeScalar.setup_exact h choice metadata q hq,NativeScalar.row_exact,NativeScalar.column_exact]
  simp only [NativeScalar.ideal,enc_mul,base,pointer,enc_add,tensorOffset]
  congr 1
  ring

theorem d_bound (h : HostCall) (c : Tuning) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) (step : Nat) (i : TileIndex [c.BM,c.BCS])
    (hr : FPLogical.row h c q i.1 < h.chunk) (hk : i.2.1.val < FPLogical.length h q-step*c.BCS) :
    dOffset h c q step i < alloc.extent h.Dout := by
  have hl : FPLogical.length h q ≤ h.chunk := Nat.min_le_left _ _
  have kc : step*c.BCS+i.2.1.val < h.chunk := by omega
  have kl : step*c.BCS+i.2.1.val < FPLogical.length h q := by omega
  exact (layout.bounds _ (FPGridOutput.grid_member _ _ _ hq) default).dout
    (i.1,⟨step*c.BCS+i.2.1.val,kc⟩,PUnit.unit) ⟨hr,kl⟩

theorem a_bound (h : HostCall) (c : Tuning) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) (step : Nat) (i : TileIndex [c.BCS,c.BN])
    (hk : i.1.val < FPLogical.length h q-step*c.BCS) (hn : FPLogical.column h c q i.2.1 < h.K) :
    aOffset h c q step i < alloc.extent h.A := by
  have hl : FPLogical.length h q ≤ h.chunk := Nat.min_le_left _ _
  have kc : step*c.BCS+i.1.val < h.chunk := by omega
  have kl : step*c.BCS+i.1.val < FPLogical.length h q := by omega
  exact (layout.bounds _ (FPGridOutput.grid_member _ _ _ hq) default).a
    (⟨step*c.BCS+i.1.val,kc⟩,i.2.1,PUnit.unit) ⟨kl,hn⟩

theorem residual_bound (h : HostCall) (c : Tuning) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) (i : TileIndex [c.BM,c.BN])
    (hi : FPLogical.row h c q i.1 < FPLogical.length h q ∧ FPLogical.column h c q i.2.1 < h.K)
    (hr : h.hasResidual = Bool.true) :
    let p := HostContract.parameters h
    tensorOffset h c q p.rBatch p.rRow p.rHead p.rColumn i < alloc.extent h.Residual :=
  (layout.bounds _ (FPGridOutput.grid_member _ _ _ hq) default).residual hr i hi

theorem output_bound (h : HostCall) (c : Tuning) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) (i : TileIndex [c.BM,c.BN])
    (hi : FPLogical.row h c q i.1 < FPLogical.length h q ∧ FPLogical.column h c q i.2.1 < h.K) :
    let p := HostContract.parameters h
    tensorOffset h c q p.oBatch p.oRow p.oHead p.oColumn i < alloc.extent h.Out :=
  (layout.bounds _ (FPGridOutput.grid_member _ _ _ hq) default).output i hi

#axiomsClean decode_exact
#axiomsClean dout_word
#axiomsClean a_word
#axiomsClean tensor_word
#axiomsClean d_bound
#axiomsClean a_bound
#axiomsClean residual_bound
#axiomsClean output_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeAddress
