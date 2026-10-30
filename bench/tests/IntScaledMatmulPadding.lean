import bench.tritonbench_g.int_scaled_matmul.IntScaledMatmul
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.PaddingRegression
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

def a : Region .int := ⟨"a"⟩
def b : Region .int := ⟨"b"⟩
def initial : BlockState where
  mem := fun r _ => if r="a" then .of .int 2 else if r="b" then .of .int 3 else .of .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun r _ => if r="a" then 5 else if r="b" then 7 else 0

def kernel := int_scaled_matmul_matmul_surface a b "c" 1 1 1 1 1 1 1 1 1 1 1 2 1

theorem ordinary_gemm : mmSpec initial a b 1 1 1 1 1 0 0 = 6 := by
  rfl

theorem padded_value : mmPaddedSpec initial a b 1 (mmSpan 1 2) 1 1 1 1 0 0 = 41 := by
  rw [mmPaddedSpec,ordinary_gemm]
  norm_num [mmPaddingTail,mmSpan,mmPad,initial,a,b,Region.cast,
    show ("b" : String)≠"a" from by decide]

/-- The complete source writes 41, while ordinary GEMM is 6. This is a concrete
counterexample to silently treating omitted padding as zero. -/
theorem source_executes_with_nonzero_tail :
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      final.mem "c" 0 = MemCell.of .int 41 := by
  obtain ⟨final,he,hout⟩ := int_scaled_matmul_matmul_exec_genuine a b "c"
    1 1 1 1 1 1 1 1 1 1 1 2 1 initial (by decide)
    (fun _ _ _ => Subsingleton.elim _ _)
  refine ⟨final,he,?_⟩
  have hv := hout (⟨⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit⟩ : TileIndex [1,1])
    (by decide)
  change final.mem "c" 0 = MemCell.of .int
    (mmPaddedSpec initial a b 1 (mmSpan 1 2) 1 1 1 1 0 0) at hv
  simpa only [padded_value] using hv

/-- Aligned K has an empty padding tail for every oracle. -/
theorem aligned_value (s : BlockState) (a b : Region .int)
    (K BK sam sak sbk sbn row col : Nat) (hBK : 0<BK) (hd : BK ∣ K) :
    mmPaddedSpec s a b K (mmSpan K BK) sam sak sbk sbn row col =
      mmSpec s a b K sam sak sbk sbn row col := by
  obtain ⟨n,hn⟩ := hd
  have hn' : K=n*BK := by simpa only [Nat.mul_comm] using hn
  have hdiv : (K+BK-1)/BK=n := Nat.div_eq_of_lt_le
    (by omega) (by rw [Nat.add_mul]; omega)
  simp [mmSpan,hdiv,←hn',mmPaddedSpec,mmPaddingTail]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.PaddingRegression
