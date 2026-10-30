import bench.tritonbench_g.flash_attn.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.LogicalIO
open VeriTile Triton FP.Structural FlashAttn
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem domain_eq (x : Layout) : (fun raw => RealInputs x (inputs (Option ℝ) x raw)) = Logical.RealInputs x := by
  funext raw; apply propext
  constructor
  · rintro ⟨hq,hk,hv⟩ port i
    fin_cases port
    · exact hq i.1 i.2.1 i.2.2.1 i.2.2.2.1
    · exact hk i.1 i.2.1 i.2.2.1 i.2.2.2.1
    · exact hv i.1 i.2.1 i.2.2.1 i.2.2.2.1
  · intro h
    exact ⟨fun b hd r d => h (0 : Fin 3) (b,hd,r,d,()),
      fun b hd r d => h (1 : Fin 3) (b,hd,r,d,()),fun b hd r d => h (2 : Fin 3) (b,hd,r,d,())⟩

theorem query_at (x : Layout) (raw : PublicIO.Inputs KernelReal.R x) (b h r d : Nat) :
    realAt (inputs KernelReal.R x raw).query b h r d = Logical.realAt raw 0 b h r d := by
  rfl
theorem key_at (x : Layout) (raw : PublicIO.Inputs KernelReal.R x) (b h r d : Nat) :
    realAt (inputs KernelReal.R x raw).key b h r d = Logical.realAt raw 1 b h r d := by
  rfl
theorem value_at (x : Layout) (raw : PublicIO.Inputs KernelReal.R x) (b h r d : Nat) :
    realAt (inputs KernelReal.R x raw).value b h r d = Logical.realAt raw 2 b h r d := by
  rfl

/-- The original complete grid judgment, including every Option output. -/
theorem judgment_iff (x : Layout) (scale : ℝ) (f : Inputs KernelReal.R x → Outputs KernelReal.R x) :
    (view KernelReal.algebra KernelReal.dot x (some scale) ⊨{RealInputs x} f) ↔
      (PublicIO.realIO x scale).ImplementsOn (Logical.RealInputs x)
        (fun raw => Fin.cases
          (fun i => (f (inputs KernelReal.R x raw)).values i.1 i.2.1 i.2.2.1 i.2.2.2.1)
          (Fin.cases (fun i => (f (inputs KernelReal.R x raw)).logMass i.1 i.2.1 i.2.2.1) (fun i => i.elim0))) := by
  exact Iff.of_eq (congrArg
    (fun domain => (PublicIO.realIO x scale).ImplementsOn domain
      ((view KernelReal.algebra KernelReal.dot x (some scale)).lift f)) (domain_eq x))
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.LogicalIO
