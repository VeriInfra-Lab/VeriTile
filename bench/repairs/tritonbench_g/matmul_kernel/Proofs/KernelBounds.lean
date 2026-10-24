import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelGrid
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelTensor
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelBounds
open VeriTile.Triton FP.Structural KernelSource KernelTensor
noncomputable section
set_option maxHeartbeats 3000000

theorem launched_nonempty (p : Config) (pid : Program) (hp : pid∈programs p) : 0<p.M ∧ 0<p.N := by
  have zeroceil (b : Nat) : ceil 0 b=0 := by
    by_cases hz : b=0
    · simp [ceil,hz]
    · exact Nat.div_eq_of_lt (by omega)
  have hm : p.M≠0 := by intro h;simp [programs,h,zeroceil] at hp
  have hn : p.N≠0 := by intro h;simp [programs,h,zeroceil] at hp
  omega

theorem signed (p : Prepared) (pid : Program) (hp : pid∈programs p.config) : Signed32 p.config := by
  have bounds:=(Checked.admitted_domain p.metadata p.checked
    (allocation p.a) (allocation p.b) (allocation p.c)).2
  have nonempty:=launched_nonempty p.config pid hp
  change p.config.M=0 ∨ p.config.N=0 ∨ Signed32 p.config at bounds
  rcases bounds with h|h|h
  · omega
  · omega
  · exact h

theorem a_offset (p : Config) (s : State α) (q : Nat) (i : AIndex p) :
    aOffset p s q i=p.A.addr (row p s i.1%p.M) (q*p.BK+i.2.1.val) := by
  simp only [aOffset,aRow,View.addr,BK]
  ring

theorem b_offset (p : Config) (s : State α) (q : Nat) (i : BIndex p) :
    bOffset p s q i=p.B.addr (q*p.BK+i.1.val) (col p s i.2.1%p.N) := by
  simp only [bOffset,bCol,View.addr,BK]
  ring

/-- Formed pointers, including the final otherwise-unused increment, obey
signed32 bounds. These are actual shared DSL offset expressions. -/
theorem input_formed (p : Prepared) (pid : Program) (hp : pid∈programs p.config)
    (s : State α) (q : Nat) (hq : q≤p.config.steps) :
    (∀ i : AIndex p.config,aOffset p.config (atProgram s pid) q i<2^31) ∧
    (∀ i : BIndex p.config,bOffset p.config (atProgram s pid) q i<2^31) := by
  have nonempty:=launched_nonempty p.config pid hp
  constructor
  · intro i
    rw [a_offset]
    exact source_a_formed p.config (signed p pid hp) _ _ _ nonempty.1 hq i.2.1.isLt
  · intro i
    rw [b_offset]
    exact source_b_formed p.config (signed p pid hp) _ _ _ nonempty.2 hq i.1.isLt

theorem active_loads (p : Prepared) (pid : Program) (hp : pid∈programs p.config)
    (s : State α) (q : Nat) :
    (∀ i : AIndex p.config,i.2.1.val+q*p.config.BK<p.config.K →
      aOffset p.config (atProgram s pid) q i<p.config.A.capacity) ∧
    (∀ i : BIndex p.config,i.1.val+q*p.config.BK<p.config.K →
      bOffset p.config (atProgram s pid) q i<p.config.B.capacity) := by
  have nonempty:=launched_nonempty p.config pid hp
  constructor
  · intro i hi
    rw [a_offset]
    exact source_a_capacity p.config p.valid _ _ nonempty.1 (by omega)
  · intro i hi
    rw [b_offset]
    exact source_b_capacity p.config p.valid _ _ nonempty.2 (by omega)

theorem output_formed (p : Prepared) (pid : Program) (hp : pid∈programs p.config)
    (s : State α) (i : CIndex p.config) :
    cOffset p.config (atProgram s pid) i<2^31 := by
  have h:=source_c_formed p.config (signed p pid hp) pid (i.1.val,i.2.1.val) hp
    (by simpa [lanes] using And.intro i.1.isLt i.2.1.isLt)
  simpa [cOffset,row,col,pidM,pidN,atProgram,View.addr,workOf,Nat.mul_comm,BM,BN] using h

#axiomsClean signed
#axiomsClean input_formed
#axiomsClean active_loads
#axiomsClean output_formed
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelBounds
