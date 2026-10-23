import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.LlamaFF
open VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000

def tilesM (s : Shape) := (s.M+15)/16
def tilesN (s : Shape) := (s.N+15)/16
structure Lane (s : Shape) where
  batch : Fin s.B
  tileM : Nat
  tileN : Nat
  row : Fin 16
  col : Fin 16
  mActive : tileM*16+row.val<s.M
  nActive : tileN*16+col.val<s.N
def Lane.cell (l : Lane s) : Cell s :=
  (l.batch,⟨l.tileM*16+l.row.val,l.mActive⟩,⟨l.tileN*16+l.col.val,l.nActive⟩)
def owner (x : Cell s) : Lane s where
  batch := x.1
  tileM := x.2.1.val/16
  tileN := x.2.2.val/16
  row := ⟨x.2.1.val%16,Nat.mod_lt _ (by decide)⟩
  col := ⟨x.2.2.val%16,Nat.mod_lt _ (by decide)⟩
  mActive := by dsimp; have h := x.2.1.isLt; omega
  nActive := by dsimp; have h := x.2.2.isLt; omega
theorem owner_cell (x : Cell s) : (owner x).cell=x := by
  rcases x with ⟨batch,row,col⟩
  simp only [owner,Lane.cell]
  congr 3 <;> omega
theorem lane_owner (l : Lane s) : owner l.cell=l := by
  rcases l with ⟨batch,tm,tn,row,col,hm,hn⟩
  have hr := row.isLt
  have hc := col.isLt
  simp only [owner,Lane.cell]
  congr <;> first | apply Fin.ext; simp only; omega | omega
theorem lane_unique : Function.Injective (Lane.cell : Lane s → Cell s) := by
  intro x y h
  rw [←lane_owner x,←lane_owner y,h]
theorem lane_grid (l : Lane s) :
    l.tileM<tilesM s ∧ l.tileN<tilesN s ∧
    l.tileM*tilesN s+l.tileN<tilesM s*tilesN s := by
  have hm := l.mActive
  have hn := l.nActive
  have tm : l.tileM<tilesM s := by unfold tilesM; omega
  have tn : l.tileN<tilesN s := by unfold tilesN; omega
  exact ⟨tm,tn,by nlinarith⟩
theorem flat_pid_decode (l : Lane s) :
    (l.tileM*tilesN s+l.tileN)/tilesN s=l.tileM ∧
    (l.tileM*tilesN s+l.tileN)%tilesN s=l.tileN := by
  have hn := (lane_grid l).2.1
  have positive : 0<tilesN s := by omega
  constructor
  · rw [Nat.mul_comm l.tileM,Nat.mul_add_div positive,Nat.div_eq_of_lt hn,Nat.add_zero]
  · simp [Nat.add_mod,Nat.mod_eq_of_lt hn]
theorem active_wrapped_loads (l : Lane s) :
    (l.tileM*16+l.row.val)%s.M=l.tileM*16+l.row.val ∧
    (l.tileN*16+l.col.val)%s.N=l.tileN*16+l.col.val :=
  ⟨Nat.mod_eq_of_lt l.mActive,Nat.mod_eq_of_lt l.nActive⟩

theorem source_tail_mask (K step lane : Nat) :
    lane<K-step*64 ↔ step*64+lane<K := by omega
theorem no_empty_lane (s : Shape) (empty : s.B=0 ∨ s.M=0 ∨ s.N=0) : IsEmpty (Lane s) := by
  constructor
  intro l
  rcases empty with hb|hm|hn
  · have h := l.batch.isLt; omega
  · have h := l.mActive; omega
  · have h := l.nActive; omega

#axiomsClean owner_cell
#axiomsClean lane_owner
#axiomsClean lane_unique
#axiomsClean lane_grid
#axiomsClean flat_pid_decode
#axiomsClean active_wrapped_loads
#axiomsClean source_tail_mask
end
end VeriTile.Bench.Repairs.LlamaFF
