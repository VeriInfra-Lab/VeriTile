import bench.tritonbench_g.softmax_flaggems.Proofs.FPLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.SignedAddress

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPLoop
open VeriTile.Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited α]

theorem range_int_aligned (M : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (f : Nat → β → β) (Inv : Nat → β → State α → Prop) (stop stride : Nat)
    (body_step : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      ∃ t, run M body (s.setReg idx .int [] (fun _ => (cur : Int)))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (aligned : stride∣cur) (hs : Inv cur value s) :
    ∃ t, rangeInt M idx (cur : Int) (stop : Int) stride body s=some t ∧
      Inv (walk f stop stride cur value).1 (walk f stop stride cur value).2 t := by
  obtain ⟨t,ht,_,hI⟩ := range_model_int M idx body f
    (fun cur value s=>stride∣cur ∧ Inv cur value s) stop stride
    (fun cur value s hc h=>by
      obtain ⟨t,ht,hI⟩ := body_step cur value s hc h.1 h.2
      exact ⟨t,ht,dvd_add h.1 (dvd_refl stride),hI⟩)
    cur value s ⟨aligned,hs⟩
  exact ⟨t,ht,hI⟩

theorem range_nat_aligned (M : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (f : Nat → β → β) (Inv : Nat → β → State α → Prop) (stop stride : Nat)
    (body_step : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => cur))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (aligned : stride∣cur) (hs : Inv cur value s) :
    ∃ t, range M idx cur stop stride body s=some t ∧
      Inv (walk f stop stride cur value).1 (walk f stop stride cur value).2 t := by
  obtain ⟨t,ht,_,hI⟩ := range_model M idx body f
    (fun cur value s=>stride∣cur ∧ Inv cur value s) stop stride
    (fun cur value s hc h=>by
      obtain ⟨t,ht,hI⟩ := body_step cur value s hc h.1 h.2
      exact ⟨t,ht,dvd_add h.1 (dvd_refl stride),hI⟩)
    cur value s ⟨aligned,hs⟩
  exact ⟨t,ht,hI⟩
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPLoop
