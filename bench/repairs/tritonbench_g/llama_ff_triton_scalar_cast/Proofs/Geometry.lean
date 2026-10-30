import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Output
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.GridExecution

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Grid
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (atProgram)
noncomputable section
set_option maxHeartbeats 4000000

abbrev Index (p : Parameters) := Fin p.rows × Fin p.columns
def count (p : Parameters) := ((p.rows+15)/16)*columnBlocks p
def row (p : Parameters) (pid : Nat) (i : Fin 16) := pid/columnBlocks p*16+i.val
def column (p : Parameters) (pid : Nat) (i : Fin 16) := pid%columnBlocks p*16+i.val
def address (p : Parameters) (x : Index p) := p.columns*x.1.val+x.2.val
def Owns (p : Parameters) (pid : Nat) (x : Index p) : Prop :=
  ∃ i : TileIndex [16,16], row p pid i.1=x.1.val ∧ column p pid i.2.1=x.2.val

theorem row_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : Fin 16) :
    outputRow p (atProgram s pid) i=row p pid i := rfl
theorem column_at {α : Type} (p : Parameters) (s : State α) (pid : Nat) (i : Fin 16) :
    outputColumn p (atProgram s pid) i=column p pid i := rfl

theorem address_injective (p : Parameters) : Function.Injective (address p) := by
  intro x y equal
  have hx := x.2.isLt
  have hy := y.2.isLt
  have hn : 0<p.columns := by omega
  have hc := congrArg (fun z => z%p.columns) equal
  have hr := congrArg (fun z => z/p.columns) equal
  simp [address,Nat.add_mod,Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy] at hc
  have hd (u v : Nat) (hv : v<p.columns) : (p.columns*u+v)/p.columns=u := by
    rw [Nat.mul_add_div hn,Nat.div_eq_of_lt hv,Nat.add_zero]
  simp only [address,hd _ _ hx,hd _ _ hy] at hr
  exact Prod.ext (Fin.ext hr) (Fin.ext hc)

theorem covers (p : Parameters) (x : Index p) : ∃ pid, pid<count p ∧ Owns p pid x := by
  have hx := x.1.isLt
  have hy := x.2.isLt
  have hc : 0<columnBlocks p := by unfold columnBlocks; omega
  have hb : x.2.val/16<columnBlocks p := by unfold columnBlocks; omega
  have ha : x.1.val/16<(p.rows+15)/16 := by omega
  let pid := x.1.val/16*columnBlocks p+x.2.val/16
  have hpid : pid<count p := by
    calc
      pid < x.1.val/16*columnBlocks p+columnBlocks p := Nat.add_lt_add_left hb _
      _ = (x.1.val/16+1)*columnBlocks p := by ring
      _ ≤ count p := Nat.mul_le_mul_right _ (by omega)
  refine ⟨pid,hpid,⟨⟨x.1.val%16,by omega⟩,⟨x.2.val%16,by omega⟩,PUnit.unit⟩,?_,?_⟩
  · dsimp only [row,pid]
    rw [Nat.mul_comm (x.1.val/16) (columnBlocks p),Nat.mul_add_div hc,Nat.div_eq_of_lt hb,Nat.add_zero]
    omega
  · dsimp only [column,pid]
    rw [Nat.add_mod,Nat.mul_mod_left,Nat.zero_add,Nat.mod_eq_of_lt hb]
    simp only [Nat.mod_eq_of_lt hb]
    omega

theorem active_index {α : Type} (p : Parameters) (s : State α) (pid : Nat)
    (i : TileIndex [16,16]) (h : active p (atProgram s pid) i) :
    ∃ x : Index p, Owns p pid x ∧
      outputOffset p (atProgram s pid) i=address p x := by
  exact ⟨(⟨row p pid i.1,h.1⟩,⟨column p pid i.2.1,h.2⟩),⟨i,rfl,rfl⟩,rfl⟩

theorem empty (p : Parameters) (h : p.rows=0) : count p=0 := by
  simp [count,h]

#axiomsClean address_injective
#axiomsClean covers
#axiomsClean active_index
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Grid
