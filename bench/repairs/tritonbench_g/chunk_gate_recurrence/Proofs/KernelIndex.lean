import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelIndex
open KernelProgram
noncomputable section

theorem mixed (a b n m : Nat) (ha : a<n) (hb : b<m) : a*m+b<n*m := by
  have h := Nat.mul_le_mul_right m (Nat.succ_le_of_lt ha)
  simp only [Nat.succ_mul] at h
  omega
theorem key_inside (c : Shape) (p : Program c) (e : Fin 64) : p.key*64+e.val<c.K :=
  mixed _ _ _ _ p.keyInside e.isLt
theorem value_inside (c : Shape) (p : Program c) (j : Fin 16) : p.value*16+j.val<c.V :=
  mixed _ _ _ _ p.valueInside j.isLt
theorem matrix_address (c : Shape) (p : Program c) (t : Nat) (e : Fin 64) (j : Fin 16) :
    address c p t e.val j.val=matrixOffset c p.head t (p.key*64+e.val) (p.value*16+j.val) := by
  exact source_address c p.head t p.key p.value e.val j.val
theorem matrix_inside (c : Shape) (p : Program c) (t : Fin c.N) (e : Fin 64) (j : Fin 16) :
    address c p t.val e.val j.val<c.heads*c.N*c.K*c.V := by
  rw [matrix_address]
  exact main_bound c p.head t.val _ _ p.headInside t.isLt (key_inside c p e) (value_inside c p j)
theorem gate_inside (c : Shape) (p : Program c) (t : Fin c.N) : gateAddress c p t.val<c.heads*c.N :=
  mixed _ _ _ _ p.headInside t.isLt
theorem seed_address (c : Shape) (p : Program c) (e : Fin 64) (j : Fin 16) :
    dseedAddress c p e.val j.val=seedDenseOffset c p.head (p.key*64+e.val) (p.value*16+j.val) :=
  seed_dense_address c p.head p.key p.value e.val j.val
theorem seed_inside (c : Shape) (p : Program c) (e : Fin 64) (j : Fin 16) :
    dseedAddress c p e.val j.val<c.heads*c.K*c.V := by
  rw [seed_address]
  exact mixed _ _ _ _ (mixed _ _ _ _ p.headInside (key_inside c p e)) (value_inside c p j)
theorem gradient_address (c : Shape) (p : Program c) (t : Nat) :
    gradientAddress c p t=scratchOffset c p.head t p.key p.value := by
  unfold gradientAddress scratchOffset Shape.tiles
  ring
theorem gradient_inside (c : Shape) (p : Program c) (t : Fin c.N) :
    gradientAddress c p t.val<c.heads*c.N*c.KT*c.VT := by
  rw [gradient_address]
  have h := mixed _ _ _ _ (mixed _ _ _ _ p.headInside t.isLt)
    (mixed _ _ _ _ p.keyInside p.valueInside)
  simpa [scratchOffset,Shape.tiles,Nat.mul_assoc,Nat.add_assoc] using h

def seedExtent (c : Shape) (s : SeedStrides) :=
  (c.B-1)*s.batch+(c.H-1)*s.head+(c.K-1)*s.key+(c.V-1)*s.value+1
theorem strided_seed_inside (c : Shape) (p : Program c) (s : SeedStrides) (e : Fin 64) (j : Fin 16) :
    seedAddress c p s e.val j.val<seedExtent c s := by
  have batch : p.head/c.H<c.B := (Nat.div_lt_iff_lt_mul c.hpos).mpr p.headInside
  have head := Nat.mod_lt p.head c.hpos
  have bm := Nat.mul_le_mul_right s.batch (show p.head/c.H≤c.B-1 by omega)
  have hm := Nat.mul_le_mul_right s.head (show p.head%c.H≤c.H-1 by omega)
  have km := Nat.mul_le_mul_right s.key (show p.key*64+e.val≤c.K-1 by have := key_inside c p e; omega)
  have vm := Nat.mul_le_mul_right s.value (show p.value*16+j.val≤c.V-1 by have := value_inside c p j; omega)
  unfold seedAddress seedExtent
  omega

#axiomsClean matrix_inside
#axiomsClean seed_inside
#axiomsClean gradient_inside
#axiomsClean strided_seed_inside
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelIndex
