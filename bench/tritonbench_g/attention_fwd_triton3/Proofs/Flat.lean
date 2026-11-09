import bench.tritonbench_g.attention_fwd_triton3.Proofs.Ownership
import VeriTile.Triton.Memory.FlattenFrame

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Flat
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxHeartbeats 2000000

def address (c : Call) (A : FlatAlloc) (idx : GridIndex c.grid) (i : Lane c.BM c.D) : MemCellAddr :=
  let a:=Ownership.address c idx i
  (A.flat,A.addr a.1 a.2)

theorem output_bound (c : Call) (h : Valid c) (a : Allocated c extent)
    (idx : GridIndex c.grid) (i : Lane c.BM c.D) :
    (Ownership.address c idx i).2<extent (Ownership.address c idx i).1 := by
  cases i with
  | center r => exact lt_of_lt_of_le (carry_bound c h idx r) a.m
  | denominator r => exact lt_of_lt_of_le (carry_bound c h idx r) a.l
  | output r d => exact lt_of_lt_of_le (address_bound c.o _ _ _ _ _ _ _ _
      (batch_bound c h idx) (head_bound c h idx) (row_bound c h idx r) d.isLt) a.o

theorem member (A : FlatAlloc) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    {rg : RegionName} {off : Nat} (hb : off<A.extent rg) : rg∈A.regions := by
  by_contra hn
  rw [hc rg hn] at hb
  omega

theorem unique (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (idx jdx : GridIndex c.grid) (i j : Lane c.BM c.D)
    (he : address c A idx i=address c A jdx j) : idx=jdx ∧ i=j := by
  have hi:=output_bound c h a idx i
  have hj:=output_bound c h a jdx j
  have ho:=congrArg Prod.snd he
  have hx:=A.decode_addr hd (member A hc hi) hi
  have hy:=A.decode_addr hd (member A hc hj) hj
  have hz : Ownership.address c idx i=Ownership.address c jdx j := by
    change A.addr _ _=A.addr _ _ at ho
    rw [ho,hy] at hx
    exact (Option.some.inj hx).symm
  exact Ownership.unique c h idx jdx i j hz

#axiomsClean unique
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Flat
