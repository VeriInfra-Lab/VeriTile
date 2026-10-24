import bench.repairs.tritonbench_g.matmul_kernel.Spec
namespace VeriTile.Bench.Repairs.MatmulKernel
open RealSum
set_option maxHeartbeats 3000000
noncomputable def realOps : Ops ℝ where
  zero := 0
  mma := fun _ _ _ a b acc => acc+sum (fun j => a ⟨j%8,by omega⟩*b ⟨j%8,by omega⟩) 8
  finish := id
noncomputable def product (c : Config) (m : Memory ℝ) (w : Work) (k : Nat) :=
  m .a (c.A.addr w.1 k)*m .b (c.B.addr k w.2)
noncomputable def padded (c : Config) (m : Memory ℝ) (w : Work) (k : Nat) :=
  if k<c.K then product c m w k else 0

theorem atom_real (c : Config) (m : Memory ℝ) (w : Work) (hw : c.Active w) (q j : Nat) (z : ℝ) :
    atom c realOps m w q j z=z+sum (fun e => padded c m w (q*c.BK+j*8+e)) 8 := by
  unfold atom realOps
  dsimp only
  congr 1
  apply sum_congr
  intro e he
  have em : e%8=e := Nat.mod_eq_of_lt he
  simp only [em,readA,readB,Nat.mod_eq_of_lt hw.1,Nat.mod_eq_of_lt hw.2,padded,product]
  split_ifs <;> simp

theorem atom_fold_real (c : Config) (m : Memory ℝ) (w : Work) (hw : c.Active w) (q t : Nat) (z : ℝ) :
    (List.range t).foldl (fun acc j => atom c realOps m w q j acc) z=
      z+sum (fun e => padded c m w (q*c.BK+e)) (t*8) := by
  induction t with
  | zero => simp [sum]
  | succ t ih =>
    simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil]
    rw [ih,atom_real c m w hw]
    rw [Nat.succ_mul,sum_add]
    simp only [Nat.add_assoc,add_assoc]
theorem block_real (c : Config) (hc : c.Valid) (m : Memory ℝ) (w : Work) (hw : c.Active w) (q : Nat) (z : ℝ) :
    block c realOps m w q z=z+sum (fun e => padded c m w (q*c.BK+e)) c.BK := by
  unfold block
  rw [atom_fold_real c m w hw,hc.atoms]
theorem block_fold_real (c : Config) (hc : c.Valid) (m : Memory ℝ) (w : Work) (hw : c.Active w)
    (t : Nat) (z : ℝ) :
    (List.range t).foldl (fun acc q => block c realOps m w q acc) z=z+sum (padded c m w) (t*c.BK) := by
  induction t with
  | zero => simp [sum]
  | succ t ih =>
    simp only [List.range_succ,List.foldl_append,List.foldl_cons,List.foldl_nil]
    rw [ih,block_real c hc m w hw]
    rw [Nat.succ_mul,sum_add]
    ring

theorem value_mathematical (c : Config) (hc : c.Valid) (m : Memory ℝ) (w : Work) (hw : c.Active w) :
    value c realOps m w=mathematical c m w.1 w.2 := by
  unfold value
  rw [block_fold_real c hc m w hw]
  change 0+sum (padded c m w) (c.steps*c.BK)=_
  rw [zero_add]
  change sum (fun k => if k<c.K then product c m w k else 0) (c.steps*c.BK)=_
  rw [sum_padding,Nat.min_eq_left (show c.K≤c.steps*c.BK from MatmulKernel.ceil_cover c.K c.BK hc.bk),sum_finset]
  rfl

theorem source_mathematical (c : Config) (hc : c.Valid) (m : Memory ℝ) (w : Work) (hw : c.Active w) :
    source c realOps m .c (c.C.addr w.1 w.2)=mathematical c m w.1 w.2 := by
  rw [source_run,run_readback c hc realOps m w hw,value_mathematical c hc m w hw]
#axiomsClean atom_real
#axiomsClean block_fold_real
#axiomsClean value_mathematical
#axiomsClean source_mathematical
end VeriTile.Bench.Repairs.MatmulKernel
