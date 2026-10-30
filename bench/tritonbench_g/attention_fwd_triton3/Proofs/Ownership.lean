import bench.tritonbench_g.attention_fwd_triton3.Proofs.Host

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Ownership
open VeriTile.Triton Geometry Host

inductive Lane (BM D : Nat)
  | center : Fin BM → Lane BM D
  | denominator : Fin BM → Lane BM D
  | output : Fin BM → Fin D → Lane BM D

def active (c : Call) : Lane c.BM c.D → Prop
  | .center _ => True
  | .denominator _ => c.finish=0
  | .output _ _ => True

def address (c : Call) (idx : GridIndex c.grid) : Lane c.BM c.D → MemCellAddr
  | .center r => ("M",c.carryAddress idx r.val)
  | .denominator r => ("L",c.carryAddress idx r.val)
  | .output r d => ("O",c.outAddress idx r.val d.val)

theorem index_eq (c : Call) (i j : GridIndex c.grid)
    (hp : c.tile i=c.tile j) (hh : c.hz i=c.hz j) : i=j := by
  funext ax
  have hax : ax.val<2 := ax.isLt
  fin_cases ax
  · exact Fin.ext hp
  · exact Fin.ext hh

theorem row_injective (c : Call) (h : Valid c) (i j : GridIndex c.grid)
    (r s : Fin c.BM) (he : c.row i r.val=c.row j s.val) : c.tile i=c.tile j ∧ r=s := by
  have he':=chunk_unique r.isLt s.isLt he
  exact ⟨he'.1,Fin.ext he'.2⟩

theorem carry_injective (c : Call) (h : Valid c) (i j : GridIndex c.grid)
    (r s : Fin c.BM) (he : c.carryAddress i r.val=c.carryAddress j s.val) : i=j ∧ r=s := by
  have he':=chunk_unique (row_bound c h i r) (row_bound c h j s) he
  have hr:=row_injective c h i j r s he'.2
  exact ⟨index_eq c i j hr.1 he'.1,hr.2⟩

theorem output_injective (c : Call) (h : Valid c) (i j : GridIndex c.grid)
    (r s : Fin c.BM) (d e : Fin c.D)
    (he : c.outAddress i r.val d.val=c.outAddress j s.val e.val) :
    i=j ∧ r=s ∧ d=e := by
  have hv:=address_injective c.o c.H c.rounded c.D h.output
    _ _ _ _ _ _ _ _ (head_bound c h i) (row_bound c h i r) d.isLt
    (head_bound c h j) (row_bound c h j s) e.isLt he
  have hi:=Nat.div_add_mod (c.hz i) c.H
  have hj:=Nat.div_add_mod (c.hz j) c.H
  have hhz : c.hz i=c.hz j := by rw [hv.1,hv.2.1] at hi; omega
  have hr:=row_injective c h i j r s hv.2.2.1
  exact ⟨index_eq c i j hr.1 hhz,hr.2,Fin.ext hv.2.2.2⟩

theorem unique (c : Call) (h : Valid c) (i j : GridIndex c.grid)
    (x y : Lane c.BM c.D) (he : address c i x=address c j y) : i=j ∧ x=y := by
  cases x <;> cases y <;> simp [address] at he
  · obtain ⟨hi,hr⟩:=carry_injective c h i j _ _ he
    exact ⟨hi,by rw [hr]⟩
  · obtain ⟨hi,hr⟩:=carry_injective c h i j _ _ he
    exact ⟨hi,by rw [hr]⟩
  · obtain ⟨hi,hr,hd⟩:=output_injective c h i j _ _ _ _ he
    exact ⟨hi,by rw [hr,hd]⟩

#print axioms unique
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Ownership
