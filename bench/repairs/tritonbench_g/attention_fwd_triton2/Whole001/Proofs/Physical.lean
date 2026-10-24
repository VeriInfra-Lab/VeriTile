import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.Access
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Physical
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
variable {α : Type} [Inhabited α]

/-- Finite typed word storage. FP32 input words and BF16 output words have
four/two-byte ranges in Machine.Domain. This is not a raw-byte endian model. -/
structure Store (α : Type) where
  capacity : RegionName → Nat
  cells : (r : RegionName) → Fin (capacity r) → Cell α
  outside : Cell α

def Store.extend (m : Store α) : State α where
  mem:=fun r off=>if h : off<m.capacity r then m.cells r ⟨off,h⟩ else m.outside
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun d _ _=>defaultValue d

def restrict (capacity : RegionName → Nat) (s : State α) : Store α :=
  ⟨capacity,(fun r i=>s.mem r i.val),.mk .real default⟩

structure InputFits (p : Parameters) (m : Store α) : Prop where
  query : words p≤m.capacity p.Q
  key : words p≤m.capacity p.K
  value : words p≤m.capacity p.V
  queryScale : p.batches*p.heads*((p.length+127)/128)≤m.capacity p.QS
  keyScale : p.batches*p.heads*((p.length+63)/64)≤m.capacity p.KS

/-- Bounds against the actual finite input allocations, derived from the
canonical source masks and InputFits rather than assumed as execution facts. -/
structure InputAccess (p : Parameters) (m : Store α) : Prop where
  query : ∀ x<(p.length+127)/128,∀ y<p.batches*p.heads,∀ r<128,∀ d<128,
    x*128+r<p.length → d<96 →
      offset p.q (y/p.heads) (y%p.heads) (x*128+r) d<m.capacity p.Q
  keyValue : ∀ y<p.batches*p.heads,∀ row<p.length,∀ n<64,∀ d<128,
    n<p.length-row → d<96 →
      offset p.k (y/p.heads) (y%p.heads) (row+n) d<m.capacity p.K ∧
      offset p.v (y/p.heads) (y%p.heads) (row+n) d<m.capacity p.V
  queryScale : ∀ x<(p.length+127)/128,∀ y<p.batches*p.heads,
    y*((p.length+127)/128)+x<m.capacity p.QS
  keyScale : ∀ y<p.batches*p.heads,∀ row<p.length,
    y*((p.length+63)/64)+row/64<m.capacity p.KS

theorem input_access (p : Parameters) (m : Store α) (fits : InputFits p m)
    (canon : Canonical p) : InputAccess p m := by
  have safe:=Access.safe p canon
  refine ⟨?_,?_,?_,?_⟩
  · intro x hx y hy r hr d hd hm hf
    exact lt_of_lt_of_le (safe.query x hx y hy r hr d hd hm hf) fits.query
  · intro y hy row hrow n hn d hd hn' hd'
    have h:=safe.keyValue y hy row hrow n hn d hd hn' hd'
    exact ⟨lt_of_lt_of_le h.1 fits.key,lt_of_lt_of_le h.2 fits.value⟩
  · intro x hx y hy
    exact lt_of_lt_of_le (safe.queryScale x hx y hy) fits.queryScale
  · intro y hy row hrow
    exact lt_of_lt_of_le (safe.keyScale y hy row hrow) fits.keyScale

def afterCapacity (p : Parameters) (m : Store α) (unused : RegionName) : RegionName → Nat :=
  fun r=>if r=p.Out then words p else if r=unused then p.batches*p.heads*p.length else m.capacity r

structure Returned (α : Type) where
  tensor : Descriptor
  memory : Store α

/-- Runs the actual source grid from current memory, then restricts the proven
bounded result to finite allocated word regions. No return value is assumed. -/
def invoke (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters)
    (abi : InputABI p) (before : Store α) (alloc : Allocation p abi before.extend)
    (work : List Program) : Option (Returned α) :=
  (Whole.invoke M dot p abi before.extend alloc work).map fun result=>
    ⟨result.tensor,restrict (afterCapacity p before alloc.unusedM.region) result.state⟩

theorem allocation_separated (p : Parameters) (abi : InputABI p) (before : State α)
    (alloc : Allocation p abi before) :
    p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS := by
  have hq:=(alloc.fresh.2 abi.query (by simp)).2.2.1
  have hk:=(alloc.fresh.2 abi.key (by simp)).2.2.1
  have hv:=(alloc.fresh.2 abi.value (by simp)).2.2.1
  have hqs:=(alloc.fresh.2 abi.queryScale (by simp)).2.2.1
  have hks:=(alloc.fresh.2 abi.keyScale (by simp)).2.2.1
  rw [abi.regions.1] at hq
  rw [abi.regions.2.1] at hk
  rw [abi.regions.2.2.1] at hv
  rw [abi.regions.2.2.2.1] at hqs
  rw [abi.regions.2.2.2.2] at hks
  exact ⟨hq,hk,hv,hqs,hks⟩

theorem finite_return (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters)
    (bases : Bases) (domain : Domain p bases) (abi : InputABI p)
    (before : Store α) (fits : InputFits p before) (alloc : Allocation p abi before.extend)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M dot p abi before alloc work=some result ∧ result.tensor=alloc.output ∧
      result.memory.capacity p.Out=words p ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),
          index.val=logicalAddress p b h n d ∧ result.memory.cells p.Out index=.mk .bf16
            (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
              (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd))) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ InputAccess p before := by
  let storage:=Machine.storage p domain.canonical (allocation_separated p abi before.extend alloc)
  obtain ⟨ret,hr,ht,hv,hf⟩:=host_result M dot p storage abi before.extend alloc work legal
  let result : Returned α:=⟨ret.tensor,restrict (afterCapacity p before alloc.unusedM.region) ret.state⟩
  refine ⟨result,by simp [invoke,hr,result],ht,by simp [result,restrict,afterCapacity],?_,?_,
    Access.safe p domain.canonical,Machine.closed p bases domain,
    input_access p before fits domain.canonical⟩
  · intro b hb h hh n hn d hd
    have bound:=storage.output b hb h hh n hn d hd
    change logicalAddress p b h n d<words p at bound
    have bound' : logicalAddress p b h n d<result.memory.capacity p.Out := by
      simpa [result,restrict,afterCapacity] using bound
    refine ⟨⟨logicalAddress p b h n d,bound'⟩,rfl,?_⟩
    exact hv b hb h hh n hn d hd
  · intro r i ho hm
    have cap : result.memory.capacity r=before.capacity r := by simp [result,restrict,afterCapacity,ho,hm]
    refine ⟨⟨i.val,by rw [cap];exact i.isLt⟩,rfl,?_⟩
    change ret.state.mem r i.val=before.cells r i
    rw [hf r i.val ho hm]
    simp [Store.extend,i.isLt]

#axiomsClean input_access
#axiomsClean finite_return
#specNonCircular Store avoiding [FP.Structural.exec,Whole.invoke]
#specNonCircular InputFits avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Physical
