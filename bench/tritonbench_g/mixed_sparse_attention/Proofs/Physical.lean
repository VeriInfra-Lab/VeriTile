import bench.tritonbench_g.mixed_sparse_attention.Proofs.Host
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Physical
open VeriTile Triton FP.Structural Source Whole
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000

/-- Finite allocations before the host creates its fresh output. Unallocated
reads are only a total-model extension; access bounds are proved separately. -/
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

def allocatedCapacity (c : Config) (before : Store α) (r : RegionName) : Nat :=
  if r=c.out then tensorSize c else before.capacity r

def restrict (capacity : RegionName → Nat) (outside : Cell α) (s : State α) : Store α :=
  ⟨capacity,(fun r i=>s.mem r i.val),outside⟩

structure Returned (α : Type) where
  memory : Store α
  result : Tensor

def invoke (M : Program → Algebra α) (D : Program → Dot α) (c : Config) (zero : α)
    (before : Store α) (work : List Program) : Option (Returned α) :=
  (Whole.invoke M D c zero before.extend work).map fun result=>
    ⟨restrict (allocatedCapacity c before) before.outside result.memory,result.result⟩

/-- Fresh output allocation, finite returned logical cells, and preservation of
every previously allocated cell. Read-only input aliases remain permitted. -/
theorem finite_return (M : Program → Algebra α) (D : Program → Dot α) (c : Config) (zero : α)
    (before : Store α) (fresh : before.capacity c.out=0)
    (sep : Separated c) (layout : Contiguous c) (hl : LengthsBound c before.extend)
    (hbm : 0<c.BM) (hbn : 0<c.BN) (work : List Program)
    (legal : (programs (grid c)).Perm work) :
    ∃ result,invoke M D c zero before work=some result ∧ result.result=returnedTensor c ∧
      result.memory.capacity=allocatedCapacity c before ∧
      (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
        ∃ index : Fin (result.memory.capacity c.out),index.val=logicalAddress c hz n d ∧
          result.memory.cells c.out index=.mk .real (ofFloat .real
            (if n<(before.extend.mem (Region.cast c.seq) (hz/c.H)).read .nat then
              Whole.answer (M (n/c.BM,hz,0)) (D (n/c.BM,hz,0)) c
                (start (grid c) (n/c.BM,hz,0) before.extend) (lane c hbm n d hd)
             else zero))) ∧
      (∀ r (i : Fin (before.capacity r)),∃ j : Fin (result.memory.capacity r),
        j.val=i.val ∧ result.memory.cells r j=before.cells r i) := by
  obtain ⟨ret,hr,hdesc,hv,hframe⟩:=returned_source M D c zero before.extend sep layout hl hbm hbn work legal
  let result : Returned α:=⟨restrict (allocatedCapacity c before) before.outside ret.memory,ret.result⟩
  refine ⟨result,by simp [invoke,hr,result],hdesc,rfl,?_,?_⟩
  · intro hz hh n hn d hd
    have bound:=dense_bound (c.Z*c.H) c.T c.BD hz n d hh hn hd
    have fits : logicalAddress c hz n d<result.memory.capacity c.out := by
      simpa [result,restrict,allocatedCapacity,tensorSize,logicalAddress] using bound
    refine ⟨⟨logicalAddress c hz n d,fits⟩,rfl,?_⟩
    exact hv hz hh n hn d hd
  · intro r i
    have hout : r≠c.out := by
      intro he
      have h:=i.isLt
      have hz : before.capacity r=0 := by rw [he,fresh]
      have impossible := Nat.lt_of_lt_of_eq h hz
      omega
    have fits : i.val<result.memory.capacity r := by
      simpa [result,restrict,allocatedCapacity,hout] using i.isLt
    refine ⟨⟨i.val,fits⟩,rfl,?_⟩
    change ret.memory.mem r i.val=before.cells r i
    rw [hframe r i.val (Or.inl hout)]
    simp [Store.extend,i.isLt]

#axiomsClean finite_return
#specNonCircular Store avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Physical
