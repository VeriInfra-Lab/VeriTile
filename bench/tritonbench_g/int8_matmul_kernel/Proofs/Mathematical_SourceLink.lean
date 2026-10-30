import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Grid
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Inverse of the actual store lane indexing, for each fixed source pid.
This completes the active-lane reindexing used by programCode. -/
theorem source_store_inverse (s : Shape) (b : Buffers) (c : Fin 15) (pid : Nat) (p : Address)
    (write : writes s b c pid p) :
    ∃ i j,i<(configs c).bm ∧ j<(configs c).bn ∧
      row s c pid i<s.m ∧ col s c pid j<s.n ∧
      outputOffset s (row s c pid i) (col s c pid j)=p.2 := by
  have hn : 0<s.n := by have := write.2.1; nlinarith
  have maxn : max s.n 1=s.n := max_eq_left hn
  have bm : 0<(configs c).bm := by have := config_bounds c; omega
  have bn : 0<(configs c).bn := by have := config_bounds c; omega
  have hrow : p.2/max s.n 1<s.m := by
    rw [maxn]
    exact (Nat.div_lt_iff_lt_mul hn).mpr write.2.1
  have hcol : p.2%max s.n 1<s.n := by rw [maxn]; exact Nat.mod_lt _ hn
  have hr : row s c pid ((p.2/max s.n 1)%(configs c).bm)=p.2/max s.n 1 := by
    simp only [row,write.2.2.1]
    simpa only [Nat.mul_comm] using Nat.div_add_mod (p.2/max s.n 1) (configs c).bm
  have hc : col s c pid ((p.2%max s.n 1)%(configs c).bn)=p.2%max s.n 1 := by
    simp only [col,write.2.2.2]
    simpa only [Nat.mul_comm] using Nat.div_add_mod (p.2%max s.n 1) (configs c).bn
  refine ⟨(p.2/max s.n 1)%(configs c).bm,(p.2%max s.n 1)%(configs c).bn,
    Nat.mod_lt _ bm,Nat.mod_lt _ bn,?_,?_,?_⟩
  · simpa only [hr] using hrow
  · simpa only [hc] using hcol
  · simp only [outputOffset,hr,hc]
    simpa only [Nat.mul_comm] using Nat.div_add_mod p.2 (max s.n 1)

theorem active_wrapped_gathers (s : Shape) (c : Fin 15) (pid i j : Nat)
    (hm : row s c pid i<s.m) (hn : col s c pid j<s.n) :
    row s c pid i%s.m=row s c pid i ∧ col s c pid j%s.n=col s c pid j := by
  exact ⟨Nat.mod_eq_of_lt hm,Nat.mod_eq_of_lt hn⟩

theorem active_source_accumulation (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (legal : s.legal c) (memory : Memory α) (pid i j : Nat)
    (hm : row s c pid i<s.m) (hn : col s c pid j<s.n) :
    sourceAccumulation a s b c memory (row s c pid i%s.m) (col s c pid j%s.n)=
      callerAccumulation a s b c memory (row s c pid i) (col s c pid j) := by
  rw [(active_wrapped_gathers s c pid i j hm hn).1,(active_wrapped_gathers s c pid i j hm hn).2,
    sourceAccumulation_eq a s b c legal]

#axiomsClean source_store_inverse
#axiomsClean active_source_accumulation
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
