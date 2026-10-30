import bench.tritonbench_g.int4_matmul.IO

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution KernelSource
noncomputable section
set_option maxHeartbeats 3000000

abbrev row (x : Raw.Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (Raw.parameters x t)) :=
  rowBlock (Raw.parameters x t) s*t.bm+i.1.val
abbrev column (x : Raw.Arguments) (t : Tile) (s : RegisterState α) (i : CIndex (Raw.parameters x t)) :=
  columnBlock (Raw.parameters x t) s*t.bn+i.2.1.val

theorem output_address (x : Raw.Arguments) (t : Tile) (s : RegisterState α)
    (i : CIndex (Raw.parameters x t)) :
    outputOffset (Raw.parameters x t) s i=row x t s i*x.shape.n+column x t s i := by
  simp only [outputOffset,Raw.parameters,Nat.one_mul,row,column,Nat.mul_comm]

theorem active_unique (x : Raw.Arguments) (t : Tile) (s : RegisterState α)
    (i j : CIndex (Raw.parameters x t)) (hi : active (Raw.parameters x t) s i)
    (hj : active (Raw.parameters x t) s j)
    (same : outputOffset (Raw.parameters x t) s j=outputOffset (Raw.parameters x t) s i) : j=i := by
  rw [output_address,output_address] at same
  have both := address_injective x.shape ⟨row x t s j,column x t s j,0⟩
    ⟨row x t s i,column x t s i,0⟩ hj.2 hi.2 same
  have hr : j.1.val=i.1.val := by have h := both.1;dsimp [row] at h;omega
  have hc : j.2.1.val=i.2.1.val := by have h := both.2;dsimp [column] at h;omega
  rcases i with ⟨a,b,⟨⟩⟩
  rcases j with ⟨c,d,⟨⟩⟩
  simp only [Prod.mk.injEq,Fin.ext_iff]
  exact ⟨hr,hc,True.intro⟩

theorem invocation_row (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (p : IO.Program) (s : RegisterState (SourceGraph.Node α)) :
    rowBlock (Raw.parameters x t) ((IO.invocation x t o rest p).initial s)=
      GroupedGrid.row (blocksM x.shape t) (blocksN x.shape t) t.gm p.1 := rfl

theorem invocation_column (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (p : IO.Program) (s : RegisterState (SourceGraph.Node α)) :
    columnBlock (Raw.parameters x t) ((IO.invocation x t o rest p).initial s)=
      GroupedGrid.col (blocksM x.shape t) (blocksN x.shape t) t.gm p.1 := rfl

/-- A listed CTA owns exactly the local index selected by coordinate remainders. -/
theorem owner_index (x : Raw.Arguments) (t : Tile) (ht : t.Valid) (o : Ops α)
    (rest : Algebra α) (p : IO.Program) (s : RegisterState (SourceGraph.Node α))
    (r c : Nat) (own : IO.owns x.shape t p r c) :
    ∃ i : CIndex (Raw.parameters x t),
      row x t ((IO.invocation x t o rest p).initial s) i=r ∧
      column x t ((IO.invocation x t o rest p).initial s) i=c := by
  refine ⟨(⟨r%t.bm,Nat.mod_lt _ ht.bmpos⟩,⟨c%t.bn,Nat.mod_lt _ ht.bnpos⟩,()),?_,?_⟩
  · change rowBlock _ _ *t.bm+r%t.bm=r
    rw [invocation_row,←own.1,Nat.div_add_mod']
  · change columnBlock _ _ *t.bn+c%t.bn=c
    rw [invocation_column,←own.2,Nat.div_add_mod']

theorem source_owns (x : Raw.Arguments) (t : Tile) (ht : t.Valid) (o : Ops α)
    (rest : Algebra α) (p : IO.Program) (s : RegisterState (SourceGraph.Node α))
    (i : CIndex (Raw.parameters x t)) :
    IO.owns x.shape t p
      (row x t ((IO.invocation x t o rest p).initial s) i)
      (column x t ((IO.invocation x t o rest p).initial s) i) := by
  simp only [IO.owns,row,column,div_coordinate _ _ _ ht.bmpos i.1.isLt,
    div_coordinate _ _ _ ht.bnpos i.2.1.isLt,invocation_row,invocation_column,and_self]
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
