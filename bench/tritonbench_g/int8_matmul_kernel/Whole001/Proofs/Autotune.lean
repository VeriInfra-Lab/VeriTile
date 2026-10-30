import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Grid
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

theorem run_absorbs (a : Arithmetic α β) (s : Shape) (b : Buffers) (c trial : Fin 15)
    (legal : s.legal c) (trialLegal : s.legal trial) (sep : Separate b) (entry : Memory α) :
    run a s b c (run a s b trial entry)=run a s b c entry := by
  have reads (p : Address) (read : b.reads p) :
      run a s b trial entry p=entry p :=
    run_frame a s b trial trialLegal sep entry p (fun write => separate_read_write b sep p read write.1)
  funext p
  by_cases active : p.1=b.outRegion ∧ p.2<s.m*s.n
  · have hn : 0<s.n := by have := active.2; nlinarith
    have maxn : max s.n 1=s.n := max_eq_left hn
    have hm : p.2/max s.n 1<s.m := by rw [maxn]; exact (Nat.div_lt_iff_lt_mul hn).mpr active.2
    have hc : p.2%max s.n 1<s.n := by rw [maxn]; exact Nat.mod_lt _ hn
    have off : outputOffset s (p.2/max s.n 1) (p.2%max s.n 1)=p.2 := by
      unfold outputOffset
      simpa only [Nat.mul_comm] using Nat.div_add_mod p.2 (max s.n 1)
    have addr : outAddress s b (p.2/max s.n 1) (p.2%max s.n 1)=p := by
      unfold outAddress
      rw [off,←active.1]
    rw [←addr,run_output a s b c legal sep _ _ _ hm hc,
      run_output a s b c legal sep _ _ _ hm hc,
      accumulation_reads a s b c entry _ reads]
  · rw [run_frame a s b c legal sep _ p active,
      run_frame a s b trial trialLegal sep entry p active,
      run_frame a s b c legal sep entry p active]

def trialRun (a : Arithmetic α β) (s : Shape) (b : Buffers) 
    (trials : List (Fin 15)) (entry : Memory α) : Memory α :=
  Schedule.run (fun c => run a s b c) trials entry

theorem trial_absorbed (a : Arithmetic α β) (s : Shape) (b : Buffers) 
    (trials : List (Fin 15)) (c : Fin 15) (legal : s.legal c) (trialsLegal : ∀ c∈trials,s.legal c) (sep : Separate b) (entry : Memory α) :
    run a s b c (trialRun a s b trials entry)=run a s b c entry := by
  induction trials generalizing entry with
  | nil => rfl
  | cons first tail ih =>
    change run a s b c (trialRun a s b tail (run a s b first entry))=_
    rw [ih (by intro q hq; exact trialsLegal q (List.mem_cons_of_mem first hq)),
      run_absorbs a s b c first legal (trialsLegal first (by simp)) sep entry]

#axiomsClean run_absorbs
#axiomsClean trial_absorbed
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
