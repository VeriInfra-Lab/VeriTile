import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawModel

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- These facts follow from the two original host configurations and a legal
program id. They are not assumptions about execution or floating values. -/
theorem launch_parameters {α : Type} (A B C : RegionName) (st : Storage)
    (M N K sam sak sbk sbn : Nat) (s : State α)
    (hp : s.pids 0 < ((M+127)/128) * ((N+255)/256)) :
    let p := SourceScope.parameters A B C st M N K sam sak sbk sbn
    (control p s).pidM = (pidM p s : Int) ∧
    (control p s).pidN = (pidN p s : Int) ∧ blockCount p = p.numKBlocks := by
  let p := SourceScope.parameters A B C st M N K sam sak sbk sbn
  have hm : signedCeil p.M (BM p) = (blocksM p : Int) := by
    simpa [p,SourceScope.parameters,BM,blocksM] using (signedCeil_configs M).2.1
  have hn : signedCeil p.N (BN p) = (blocksN p : Int) := by
    simpa [p,SourceScope.parameters,BN,blocksN] using (signedCeil_configs N).2.2
  have hf : firstM p s ≤ blocksM p := by
    have hc : 0 < (N+255)/256 := by
      by_contra hn
      have hz : (N+255)/256 = 0 := by omega
      simpa only [hz,Nat.mul_zero,Nat.not_lt_zero] using hp
    have hd := Nat.div_mul_le_self (s.pids 0) (8*((N+255)/256))
    change s.pids 0 / (8*((N+255)/256)) * 8 ≤ (M+127)/128
    nlinarith
  obtain ⟨hpm,hpn⟩ := control_coordinates p s hm hn hf
  refine ⟨hpm,hpn,?_⟩
  cases st
  · simpa [p,SourceScope.parameters,SourceScope.blockK,BK,blockCount] using
      congrArg Int.toNat (signedCeil_configs K).1
  · simpa [p,SourceScope.parameters,SourceScope.blockK,BK,blockCount] using
      congrArg Int.toNat (signedCeil_configs K).2.1

#axiomsClean launch_parameters
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
