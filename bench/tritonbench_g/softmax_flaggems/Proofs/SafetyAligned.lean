import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyBase
import bench.tritonbench_g.softmax_flaggems.Proofs.SignedLoops

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
open VeriTile Triton FP.Structural Safety
noncomputable section
variable {α β : Type} [Inhabited α]

theorem range_safe_int_aligned (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (Inv : Nat → β → State α → Prop)
    (f : Nat → β → β) (stop stride : Nat)
    (safe : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      traceSafeList M bounds body (s.setReg idx .int [] (fun _ => (cur : Int))))
    (next : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      ∃ t,run M body (s.setReg idx .int [] (fun _ => (cur : Int)))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (aligned : stride∣cur) (hs : Inv cur value s) :
    rangeIntSafe M bounds idx (cur : Int) (stop : Int) stride body s := by
  apply range_safe_int M bounds idx body
    (fun cur value s=>stride∣cur ∧ Inv cur value s) f stop stride
    (fun cur value s hc h=>safe cur value s hc h.1 h.2)
    (fun cur value s hc h=>by
      obtain ⟨t,ht,hI⟩ := next cur value s hc h.1 h.2
      exact ⟨t,ht,dvd_add h.1 (dvd_refl stride),hI⟩)
    cur value s ⟨aligned,hs⟩

theorem range_safe_nat_aligned (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (Inv : Nat → β → State α → Prop)
    (f : Nat → β → β) (stop stride : Nat)
    (safe : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => cur)))
    (next : ∀ cur value s, cur<stop → stride∣cur → Inv cur value s →
      ∃ t,run M body (s.setReg idx .nat [] (fun _ => cur))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (aligned : stride∣cur) (hs : Inv cur value s) :
    rangeSafe M bounds idx cur stop stride body s := by
  apply range_safe M bounds idx body
    (fun cur value s=>stride∣cur ∧ Inv cur value s) f stop stride
    (fun cur value s hc h=>safe cur value s hc h.1 h.2)
    (fun cur value s hc h=>by
      obtain ⟨t,ht,hI⟩ := next cur value s hc h.1 h.2
      exact ⟨t,ht,dvd_add h.1 (dvd_refl stride),hI⟩)
    cur value s ⟨aligned,hs⟩
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
