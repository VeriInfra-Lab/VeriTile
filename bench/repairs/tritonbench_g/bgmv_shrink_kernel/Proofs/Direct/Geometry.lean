import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Kernels

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

structure Program (h : Call) where
  batch : Fin h.batches
  split : Fin 64

def rank (split c e : Nat) := c*16384+(e+split*256)
def inputOffset (h : Call) (b k : Nat) := b*h.K+k
def weightOffset (h : Call) (slot n k : Nat) := slot*(h.N*h.K)+n*h.K+k
def outputOffset (h : Call) (b n : Nat) := b*h.columns+n

theorem covered (h : Call) : h.K ≤ h.cycles*16384 := by
  simp only [Call.cycles]
  omega

theorem padding_positive (h : Call) : 0<h.paddedK := by
  unfold Call.paddedK
  omega

theorem padded (h : Call) : h.K≤h.paddedK := by
  exact (covered h).trans (Nat.mul_le_mul_right 16384 (Nat.le_max_right 1 h.cycles))

theorem rank_inside (h : Call) (s : Fin 64) (c : Fin h.cycles) (e : Fin 256) :
    rank s.val c.val e.val < h.paddedK := by
  have := s.isLt
  have := c.isLt
  have := e.isLt
  have := Nat.le_max_right 1 h.cycles
  unfold rank Call.paddedK
  omega

/-- Every K coordinate belongs to exactly one split, cycle and rank lane. -/
theorem rank_coverage (h : Call) (k : Fin h.K) :
    ∃! t : Fin 64 × Fin h.cycles × Fin 256,rank t.1.val t.2.1.val t.2.2.val=k.val := by
  have bound := k.isLt
  have capacity := covered h
  let s : Fin 64 := ⟨k.val%16384/256,by omega⟩
  let c : Fin h.cycles := ⟨k.val/16384,by omega⟩
  let e : Fin 256 := ⟨k.val%256,by omega⟩
  refine ⟨(s,c,e),?_,?_⟩
  · dsimp [rank,s,c,e]
    omega
  · intro t same
    obtain ⟨a,b,d⟩ := t
    have ha := a.isLt
    have hd := d.isLt
    have hc := b.isLt
    dsimp only [rank] at same
    apply Prod.ext
    · apply Fin.ext
      dsimp [s]
      omega
    · apply Prod.ext
      · apply Fin.ext
        dsimp [c]
        omega
      · apply Fin.ext
        dsimp [e]
        omega

theorem output_injective (h : Call) (b d : Nat) (n m : Fin h.N)
    (same : outputOffset h b n.val=outputOffset h d m.val) : b=d ∧ n=m := by
  have bn := n.isLt.trans_le h.outputCovered
  have bm := m.isLt.trans_le h.outputCovered
  have positive : 0<h.columns := by omega
  unfold outputOffset at same
  have rows := congrArg (fun i => i/h.columns) same
  dsimp only at rows
  rw [Nat.add_comm (b*h.columns),Nat.add_comm (d*h.columns),
    Nat.add_mul_div_right _ _ positive,Nat.add_mul_div_right _ _ positive,
    Nat.div_eq_of_lt bn,Nat.div_eq_of_lt bm] at rows
  simp only [Nat.zero_add] at rows
  exact ⟨rows,Fin.ext (by simpa only [rows,add_right_inj] using same)⟩

#axiomsClean rank_coverage
#axiomsClean output_injective
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
