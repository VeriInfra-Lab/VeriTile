import bench.tritonbench_g.log_softmax.Proofs.Execution

namespace VeriTile.Bench.Repairs.LogSoftmax.Grid
open Address
noncomputable section
open scoped Classical

def groups (M BM : Nat) : Nat := (M+BM-1)/BM

def entries (M K BM C : Nat) : List (Nat × Nat × Nat × Nat) :=
  (List.range (groups M BM)).flatMap fun p=>
  (List.range K).flatMap fun k=>
  (List.range BM).flatMap fun i=>
  (List.range C).map fun j=>(p,k,i,j)

def activeCell (M N K BM : Nat) (q : Nat × Nat × Nat × Nat) : Option (Coord M N K) :=
  if hm : q.1*BM+q.2.2.1<M then
    if hn : q.2.2.2<N then
      if hk : q.2.1<K then some (⟨q.1*BM+q.2.2.1,hm⟩,⟨q.2.2.2,hn⟩,⟨q.2.1,hk⟩)
      else none
    else none
  else none

def cells (M N K BM C : Nat) : List (Coord M N K) :=
  (entries M K BM C).filterMap (activeCell M N K BM)

theorem group_cover (q : Nat) (active : q<M) (positive : 0<BM) : q/BM<groups M BM := by
  have mul : q/BM*BM≤q:=Nat.div_mul_le_self q BM
  have h : (q/BM+1)*BM≤M+BM-1 := by
    rw [Nat.add_mul,one_mul]
    omega
  have div := (Nat.le_div_iff_mul_le positive).mpr h
  exact Nat.lt_of_succ_le div

/-- Each active logical coordinate has its actual two-dimensional grid
program and tile lane. This is coverage of the source grid, not a kernel-result
or compiler numerical obligation. Runtime grid-size support remains separate. -/
theorem cells_complete (q : Coord M N K) (positive : 0<BM) (covers : N≤C) : q∈cells M N K BM C := by
  let raw : Nat × Nat × Nat × Nat := (q.1.val/BM,q.2.2.val,q.1.val%BM,q.2.1.val)
  have member : raw∈entries M K BM C := by
    apply List.mem_flatMap.mpr
    refine ⟨q.1.val/BM,List.mem_range.mpr (group_cover q.1.val q.1.isLt positive),?_⟩
    apply List.mem_flatMap.mpr
    refine ⟨q.2.2.val,List.mem_range.mpr q.2.2.isLt,?_⟩
    apply List.mem_flatMap.mpr
    refine ⟨q.1.val%BM,List.mem_range.mpr (Nat.mod_lt _ positive),?_⟩
    exact List.mem_map.mpr ⟨q.2.1.val,List.mem_range.mpr (lt_of_lt_of_le q.2.1.isLt covers),rfl⟩
  have row : q.1.val/BM*BM+q.1.val%BM=q.1.val:=by simpa only [Nat.mul_comm] using Nat.div_add_mod q.1.val BM
  apply List.mem_filterMap.mpr
  refine ⟨raw,member,?_⟩
  simp only [activeCell,raw,row,q.1.isLt,q.2.1.isLt,q.2.2.isLt,dif_pos]

def warps (N : Nat) : Nat := if N≤1024 then 4 else if N≤2048 then 8 else 16

def blockColumns (N : Nat) : Nat := if N≤1 then 1 else 2^(Nat.log2 (N-1)+1)

def RetainedBlock (BM : Nat) : Prop := BM=1 ∨ BM=2 ∨ BM=4 ∨ BM=8

theorem block_bounds (retained : RetainedBlock BM) : 0<BM ∧ BM≤8 := by
  rcases retained with h|h|h|h <;> omega

/-- A selected compiler schedule may order scalar stores arbitrarily. Its
only ordering contract is a permutation of the derived active source stores;
the arithmetic value of every store is defined by Execution.scalar. -/
structure Selected (M N K : Nat) where
  BM : Nat
  C : Nat
  retained : RetainedBlock BM
  columns : C=blockColumns N
  numWarps : Nat
  numWarpsBound : numWarps=warps N
  plan : Plan N C
  order : List (Coord M N K)
  stores : order.Perm (cells M N K BM C)

theorem selected_complete (s : Selected M N K) (q : Coord M N K) : q∈s.order :=
  s.stores.mem_iff.mpr (cells_complete q (block_bounds s.retained).1 s.plan.covers)

#axiomsClean group_cover
#axiomsClean cells_complete
#axiomsClean selected_complete
end
end VeriTile.Bench.Repairs.LogSoftmax.Grid
