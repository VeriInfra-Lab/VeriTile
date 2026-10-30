import bench.tritonbench_g.attention_fwd_triton3.Proofs.Whole
import VeriTile.Triton.Memory.FlattenFrame

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Flat
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
set_option maxHeartbeats 2000000

def address (c : Call) (A : FlatAlloc) (idx : GridIndex c.grid) (i : Lane c.BM c.D) : MemCellAddr :=
  let a:=Ownership.address c idx i
  (A.flat,A.addr a.1 a.2)

def source (c : Call) (A : FlatAlloc) := A.flattenKernel c.source

def writes (c : Call) (A : FlatAlloc) :=
  HostComposition.allWrites c.grid (fun _=>active c) (address c A)

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

theorem program (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (s : BlockState) (hu : s.undef=(fun _ _=>0)) (idx : GridIndex c.grid) :
    ∃ final, exec (source c A) ((A.flattenState s).withGridIndex idx)=some final ∧
      (∀ i,active c i → final.readMem (address c A idx i).1 (address c A idx i).2=Whole.result c s idx i) ∧
      BlockState.WriteWithin (HostComposition.writes c.grid (fun _=>active c) (address c A) idx)
        ((A.flattenState s).withGridIndex idx) final := by
  obtain ⟨t,he,ho,hf⟩:=Whole.program c h s idx
  refine ⟨A.flattenState t,?_,?_,?_⟩
  · have hb:=A.exec_flatten hd hc c.source (s.withGridIndex idx) (Whole.safe c h a s idx)
      (RealSource.flatten_ok c.BM c.BN c.D c.Z c.finish c.params Bool.false) hu
    rw [he,Option.map_some] at hb
    exact hb
  · intro i hi
    exact (A.flattenState_readMem hd t (member A hc (output_bound c h a idx i))
      (output_bound c h a idx i)).trans (ho i hi)
  · intro rg off hn
    symm
    apply A.flattenState_frame (s.withGridIndex idx) t
      (fun r o=>HostComposition.writes c.grid (fun _=>active c) (Ownership.address c) idx (r,o))
      (fun r o hh=>(hf r o hh).symm) rg off
    by_cases hr : rg=A.flat
    · right
      intro region offset hw heq
      obtain ⟨i,hi,hei⟩:=hw
      apply hn
      refine ⟨i,hi,?_⟩
      simp only [address,hei]
      exact Prod.ext hr.symm heq.symm
    · exact Or.inl hr

theorem complete (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (s : BlockState) (hu : s.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary (source c A) c.grid (A.flattenState s) final) ∧
      (∀ idx i,active c i → final.readMem (address c A idx i).1 (address c A idx i).2=Whole.result c s idx i) ∧
      BlockState.WriteWithin (writes c A) (A.flattenState s) final :=
  HostComposition.assemble c.grid (source c A) (A.flattenState s) (fun _=>active c) (address c A)
    (Whole.result c s) (program c h A a hd hc s hu)
    (fun i j x y _ _ he=>(unique c h A a hd hc i j x y he).1)

#print axioms complete
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Flat
