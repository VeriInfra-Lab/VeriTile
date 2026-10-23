import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.CallMachine
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.NativeSchedule

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
open VeriTile Triton Geometry Host NativeSchedule

structure Descriptor where
  (shape strides : List Nat)
  device : Nat
  (ordinary cuda fp32 resolved contiguous : Bool)

structure Arguments where
  (q k v o m l : Descriptor)

def metadataChecks (c : Call) (a : Arguments) : Bool := decide (
  (∀ t∈[a.q,a.k,a.v,a.o,a.m,a.l],t.ordinary=Bool.true ∧ t.cuda=Bool.true ∧
    t.fp32=Bool.true ∧ t.resolved=Bool.true ∧ t.device=a.q.device) ∧
  a.q.shape=[c.Z,c.H,c.NQ,c.D] ∧ a.k.shape=[c.Z,c.HK,c.NK,c.D] ∧ a.v.shape=a.k.shape ∧
  a.o.shape=[c.Z,c.H,c.rounded,c.D] ∧ a.m.shape=[c.Z,c.H,c.rounded] ∧ a.l.shape=a.m.shape ∧
  a.m.contiguous=Bool.true ∧ a.l.contiguous=Bool.true ∧
  a.q.strides=[c.q.batch,c.q.head,c.q.row,c.q.dim] ∧
  a.k.strides=[c.k.batch,c.k.head,c.k.row,c.k.dim] ∧
  a.v.strides=[c.v.batch,c.v.head,c.v.row,c.v.dim] ∧
  a.o.strides=[c.o.batch,c.o.head,c.o.row,c.o.dim] ∧
  (c.D=16 ∨ c.D=32 ∨ c.D=64 ∨ c.D=128) ∧ c.init≤1 ∧ c.finish≤1 ∧ c.window≤1 ∧ c.complement≤1)

/-- Data-pointer origins bind the word-addressed model to actual storage.
Neither alignment nor any source-dependent formed bound is assumed. -/
structure Pointers (A : FlatAlloc) where
  byteBase : RegionName → Nat
  binding : ∀ r∈A.regions,A.base r=byteBase r/4

def layoutChecks (c : Call) : Bool := decide (
  (c.BM=16 ∨ c.BM=32 ∨ c.BM=64) ∧ c.BN=c.BM ∧
  0<c.H ∧ 0<c.HK ∧ c.H%c.HK=0 ∧
  0<c.o.dim ∧ c.D*c.o.dim≤c.o.row ∧ c.rounded*c.o.row≤c.o.head ∧ c.H*c.o.head≤c.o.batch)

def machineChecks (c : Call) : Bool := decide (
  c.NQ≤2^31-65 ∧ c.NK≤2^31-65 ∧ c.rounded<2^31 ∧ c.Z*c.H≤65536 ∧
  (-(2^31)≤c.offset ∧ c.offset<2^31) ∧ (-(2^31)≤c.size ∧ c.size<2^31))

def capacityChecks (c : Call) (A : FlatAlloc) : Bool := decide (
  c.q.span c.Z c.H c.NQ c.D≤A.extent "Q" ∧
  c.k.span c.Z c.HK c.NK c.D≤A.extent "K" ∧
  c.v.span c.Z c.HK c.NK c.D≤A.extent "V" ∧
  c.o.span c.Z c.H c.rounded c.D≤A.extent "O" ∧
  c.Z*c.H*c.rounded≤A.extent "M" ∧ c.Z*c.H*c.rounded≤A.extent "L")

def separationChecks (A : FlatAlloc) : Bool := A.regions.all fun r=>A.regions.all fun s=>
  decide (r=s ∨ A.base r+A.extent r≤A.base s ∨ A.base s+A.extent s≤A.base r)

def keyRows (c : Call) : Nat :=
  if c.window≠0 ∧ c.complement=0 then
    max (c.NK+64) ((max 0 ((c.rounded : Int)+c.offset-c.size+1)).toNat+64)
  else c.NK+64

def strideExtent (c : Call) (s : Strides) (heads rows : Nat) : Nat :=
  c.Z*s.batch+heads*s.head+rows*s.row+c.D*s.dim+1

def formedExtent (c : Call) (r : RegionName) : Nat :=
  if r="Q" then strideExtent c c.q c.H (c.rounded+64)
  else if r="K" then strideExtent c c.k c.HK (keyRows c)
  else if r="V" then strideExtent c c.v c.HK (keyRows c)
  else if r="O" then strideExtent c c.o c.H (c.rounded+64)
  else c.Z*c.H*c.rounded+c.rounded+1

def pointerChecks (c : Call) (A : FlatAlloc) (p : Pointers A) : Bool :=
  decide (keyRows c<2^31) && A.regions.all fun r=>decide (
    p.byteBase r%4=0 ∧ 4*formedExtent c r<2^63 ∧ p.byteBase r+4*formedExtent c r<2^64)

theorem admittedValid (c : Call) (h : layoutChecks c=Bool.true) : Valid c := by
  simp only [layoutChecks,decide_eq_true_eq] at h
  obtain ⟨bm,bn,heads,kv,grouped,dim,row,head,batch⟩:=h
  have block : 0<c.BM ∧ c.BM∣64 := by rcases bm with h|h|h <;> rw [h] <;> decide
  exact ⟨block.1,bn ▸ block.1,block.2,heads,kv,Nat.dvd_of_mod_eq_zero grouped,⟨dim,row,head,batch⟩⟩

theorem admittedMachine (c : Call) (layout : layoutChecks c=Bool.true)
    (h : machineChecks c=Bool.true) : CallMachine.ABI c := by
  simp only [layoutChecks,decide_eq_true_eq] at layout
  simp only [machineChecks,decide_eq_true_eq] at h
  have bm : c.BM≤64 := by rcases layout.1 with h|h|h <;> omega
  exact ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,bm,by omega,h.2.2.2.2.1,h.2.2.2.2.2⟩

theorem admittedAllocated (c : Call) (A : FlatAlloc) (h : capacityChecks c A=Bool.true) : Allocated c A.extent := by
  simp only [capacityChecks,decide_eq_true_eq] at h
  exact ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2⟩

theorem admittedDisjoint (A : FlatAlloc) (h : separationChecks A=Bool.true) : A.Disjoint := by
  simp only [separationChecks,List.all_eq_true,decide_eq_true_eq] at h
  intro r hr s hs ne
  rcases h r hr s hs with he|ha|hb
  · exact False.elim (ne he)
  · exact Or.inl ha
  · exact Or.inr hb

/-- Every smaller nonnegative formed offset has aligned base, nonwrapping
absolute address, and a signed64-representable displacement. -/
theorem formed_bound (c : Call) (A : FlatAlloc) (p : Pointers A)
    (h : pointerChecks c A p=Bool.true) (r : RegionName) (hr : r∈A.regions)
    (off : Nat) (ho : off<formedExtent c r) (byte : Fin 4) :
    p.byteBase r%4=0 ∧ 4*off<2^63 ∧ 4*(A.base r+off)+byte.val<2^64 := by
  simp only [pointerChecks,Bool.and_eq_true,List.all_eq_true,decide_eq_true_eq] at h
  obtain ⟨aligned,disp,bound⟩:=h.2 r hr
  have reconstruct:=Nat.mod_add_div (p.byteBase r) 4
  rw [p.binding r hr]
  have b:=byte.isLt
  omega

theorem stride_formed (c : Call) (s : Strides) (heads rows z h n d : Nat)
    (hz : z≤c.Z) (hh : h≤heads) (hn : n≤rows) (hd : d≤c.D) :
    s.at z h n d<strideExtent c s heads rows := by
  have h0:=Nat.mul_le_mul_right s.batch hz
  have h1:=Nat.mul_le_mul_right s.head hh
  have h2:=Nat.mul_le_mul_right s.row hn
  have h3:=Nat.mul_le_mul_right s.dim hd
  simp only [Strides.at,strideExtent]
  omega

structure Safety (c : Call) (A : FlatAlloc) (p : Pointers A) : Prop where
  valid : Valid c
  machine : CallMachine.ABI c
  allocated : Allocated c A.extent
  disjoint : A.Disjoint
  pointers : ∀ r∈A.regions,∀ off<formedExtent c r,∀ byte : Fin 4,
    p.byteBase r%4=0 ∧ 4*off<2^63 ∧ 4*(A.base r+off)+byte.val<2^64

def setupChecks (c : Call) (A : FlatAlloc) (p : Pointers A) : Bool :=
  layoutChecks c && machineChecks c && capacityChecks c A && separationChecks A && pointerChecks c A p

theorem admitted (c : Call) (A : FlatAlloc) (p : Pointers A)
    (h : setupChecks c A p=Bool.true) : Safety c A p := by
  simp only [setupChecks,Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨layout,machine⟩,capacity⟩,separation⟩,pointers⟩:=h
  exact ⟨admittedValid c layout,admittedMachine c layout machine,
    admittedAllocated c A capacity,admittedDisjoint A separation,
    formed_bound c A p pointers⟩

/-- Finite native coverage stays explicitly restricted to the six accepted
profiles; arbitrary branches use the separately stated ordinary Schedule boundary. -/
def profileCheck (c : Call) (k : Kind) : Bool := decide (
  c.BM=k.block ∧ c.BN=k.block ∧ c.D=k.dim ∧ c.init=k.init ∧ c.finish=k.finish ∧
  c.window=k.window ∧ c.complement=k.complement ∧ c.params.evenM=k.even ∧ c.params.evenN=k.even)

theorem admittedProfile (c : Call) (k : Kind) (h : profileCheck c k=Bool.true) : Profile c k := by
  simp only [profileCheck,decide_eq_true_eq] at h
  exact ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2.2⟩

/-- The actual widened window, row and carry expressions have their precise
machine representation at every launched row; no coordinate equality is a premise. -/
theorem checked_machine (c : Call) (A : FlatAlloc) (p : Pointers A)
    (h : setupChecks c A p=Bool.true) (idx : GridIndex c.grid) (r : Fin c.BM) :
    (Machine.lowerNumerator (c.tile idx*c.BM) c.offset c.size).toInt=
      (c.tile idx*c.BM : Int)+c.offset-c.size+1 ∧
    (Machine.upperNumerator (c.tile idx*c.BM) c.BM c.offset c.BN).toInt=
      (c.tile idx*c.BM : Int)+c.BM-1+c.offset+c.BN ∧
    (Machine.word (c.hz idx)*Machine.word c.rounded+Machine.word (c.row idx r.val)).toInt=
      (c.hz idx : Int)*c.rounded+c.row idx r.val ∧
    (BitVec.ofInt 32 (c.row idx r.val)).toInt=(c.row idx r.val : Int) :=
  CallMachine.complete c (admitted c A p h).valid (admitted c A p h).machine idx r

open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3 in
theorem initial_lower_bound (c : Call) (valid : Valid c) (idx : GridIndex c.grid) :
    RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size<keyRows c := by
  have row := row_bound c valid idx (⟨0,valid.blockM⟩ : Fin c.BM)
  have row' : (c.tile idx*c.BM : Nat)<c.rounded := by simpa only [Call.row,Nat.add_zero] using row
  have rowInt : ((c.tile idx*c.BM : Nat) : Int)≤c.rounded := by exact_mod_cast Nat.le_of_lt row'
  simp only [RealKeys.lower,keyRows]
  split_ifs with hw
  · have lo : WindowBounds.lowerBound (c.tile idx) c.BM c.BN c.offset c.size≤
        (max 0 ((c.rounded : Int)+c.offset-c.size+1)).toNat := by
      rw [←WindowBounds.lower_bound_matches _ _ _ _ _ valid.blockN]
      apply Int.toNat_le_toNat
      have div:=Int.ediv_mul_le (((c.tile idx*c.BM : Nat) : Int)+c.offset-c.size+1)
        (show (c.BN : Int)≠0 by exact_mod_cast Nat.ne_of_gt valid.blockN)
      have mx:=le_max_right (0 : Int) ((c.rounded : Int)+c.offset-c.size+1)
      omega
    exact lt_of_lt_of_le (show WindowBounds.lowerBound (c.tile idx) c.BM c.BN c.offset c.size<
      (max 0 ((c.rounded : Int)+c.offset-c.size+1)).toNat+64 by omega) (le_max_right _ _)
  · omega

open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3 in
/-- Unlike the old active-only lemma, the new guard makes the initial INT32
block-pointer narrowing exact even when the selected window has no iteration. -/
theorem checked_initial_advance (c : Call) (A : FlatAlloc) (p : Pointers A)
    (h : setupChecks c A p=Bool.true) (idx : GridIndex c.grid) :
    (BitVec.ofInt 32 (RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size)).toInt=
      (RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size : Int) := by
  have bound:=initial_lower_bound c (admitted c A p h).valid idx
  have hp:=h
  simp only [setupChecks,Bool.and_eq_true,pointerChecks,decide_eq_true_eq] at hp
  have upper : keyRows c<2^31 := hp.2.1
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · norm_num at upper ⊢;omega

#axiomsClean initial_lower_bound
#axiomsClean checked_initial_advance

#axiomsClean stride_formed
#axiomsClean checked_machine

#axiomsClean admittedValid
#axiomsClean admittedMachine
#axiomsClean admittedAllocated
#axiomsClean admittedDisjoint
#axiomsClean formed_bound
#axiomsClean admitted
#axiomsClean admittedProfile
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
