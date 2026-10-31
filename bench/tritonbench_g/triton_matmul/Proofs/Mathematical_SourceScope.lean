import bench.tritonbench_g.triton_matmul.Sources.Mathematical_Kernels
import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPReference

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.SourceScope
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- The actual fresh row-major output is injective on active lanes even when
N is smaller than BN. This uses the existing complete typed tail proof. -/
theorem active_injective {α : Type} (A B C : RegionName) (st : Storage)
    (M N K sam sak sbk sbn : Nat) (s : State α)
    (i j : CIndex (parameters A B C st M N K sam sak sbk sbn))
    (hi : active (parameters A B C st M N K sam sak sbk sbn) s i)
    (hj : active (parameters A B C st M N K sam sak sbk sbn) s j)
    (he : cOffset (parameters A B C st M N K sam sak sbk sbn) s i=
      cOffset (parameters A B C st M N K sam sak sbk sbn) s j) : i=j := by
  let p := parameters A B C st M N K sam sak sbk sbn
  have hn : 0<max N 1 := by omega
  have hc : col p s i.2.1<max N 1 := lt_of_lt_of_le hi.2 (le_max_left _ _)
  have hc' : col p s j.2.1<max N 1 := lt_of_lt_of_le hj.2 (le_max_left _ _)
  have hoff : max N 1*row p s i.1+col p s i.2.1=max N 1*row p s j.1+col p s j.2.1 := by
    simpa [cOffset,parameters] using he
  have hm := congrArg (fun n => n%max N 1) hoff
  simp only [Nat.add_mod,Nat.mul_mod_right,zero_add,Nat.mod_eq_of_lt hc,Nat.mod_eq_of_lt hc'] at hm
  rw [hm] at hoff
  have hr := Nat.eq_of_mul_eq_mul_left hn (Nat.add_right_cancel hoff)
  have h1 : i.1=j.1 := Fin.ext (Nat.add_left_cancel hr)
  have h2 : i.2.1=j.2.1 := Fin.ext (Nat.add_left_cancel hm)
  rcases i with ⟨i,k,⟨⟩⟩; rcases j with ⟨j,l,⟨⟩⟩
  cases h1; cases h2; rfl

theorem config_table :
    blockK .half=64 ∧ stages .half=3 ∧ warps .half=8 ∧ dtype .half=.fp16 ∧
    blockK .e4m3=128 ∧ stages .e4m3=4 ∧ warps .e4m3=8 ∧ dtype .e4m3=.f8e4 := by decide

#axiomsClean config_table

#axiomsClean active_injective
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.SourceScope
