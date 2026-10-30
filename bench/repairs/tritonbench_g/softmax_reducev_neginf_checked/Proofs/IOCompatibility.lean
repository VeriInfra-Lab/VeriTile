import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOCoverage
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOReal
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IONative

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- The new logical native observation is exactly the previous ordered
per-profile word result for every old input, including all signed-word decodings. -/
theorem native_host_target (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (x : Host.Input NativeGraph.Word)
    (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) :
    nativeFormula profiles ops c (hostCounts c x) (hostInputs c x) (0 : Fin 1) (b,h,d,())=
      NativeGraph.row (NativeGraph.profilePlan (profiles ⟨b.val,h.val⟩)) ops c x ⟨b.val,h.val⟩ d.val := by
  let model := nativeModel profiles
  let obs := nativeObserve profiles ops c
  let initial := Host.launch c (mapInput NativeGraph.atom x) (fun _ => default) ⟨b.val,h.val⟩
  have bound := hostInputs_binding model NativeGraph.atom c obs [] x (fun _ => default) ⟨b.val,h.val⟩
  have row := row_logical model NativeGraph.atom c obs (hostCounts c x) []
    (hostInputs c x) initial bound (hostInputs_match c x) b h d
  have oldRow : Host.row (model ⟨b.val,h.val⟩) c (NativeGraph.nodeInput x) ⟨b.val,h.val⟩ d.val=
      rawRow model NativeGraph.atom c (hostCounts c x) (hostInputs c x) b h d := by
    simpa only [Host.row,dif_pos d.isLt] using row
  change NativeGraph.eval ops
    ((rawRow model NativeGraph.atom c (hostCounts c x) (hostInputs c x) b h d)
      ((NativeGraph.profilePlan (profiles ⟨b.val,h.val⟩)).storeReplica c.dimensions d.val))=_
  rw [←oldRow]
  rfl

/-- Existing Real normalization and value conditions imply the new domain;
its mathematical tensor is the same independent old weighted sum. -/
theorem real_host_target (c : Host.Call) (x : Host.Input (WithBot ℝ))
    (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (valid : NormalizableInputs c x logics values)
    (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) :
    mathematical c (hostCounts c x) (hostInputs c x) (0 : Fin 1) (b,h,d,())=
      (↑(SoftmaxReduceVNegInfChecked.mathematical c x logics values ⟨b.val,h.val⟩ d.val) : WithBot ℝ) := by
  change (↑(RealModel.target (x.counts b.val) (logitAt (hostInputs c x) b h)
    (fun k => (valueAt (hostInputs c x) b h k d).unbotD 0)) : WithBot ℝ)=_
  congr 1
  unfold SoftmaxReduceVNegInfChecked.mathematical RealModel.target
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    have active : k < x.counts b.val := Finset.mem_range.mp hk
    dsimp only
    rw [host_logit c x b h k active,host_value c x b h k active d,
      valid.logics ⟨b.val,h.val⟩ ⟨b.isLt,h.isLt⟩ k active,
      valid.values ⟨b.val,h.val⟩ ⟨b.isLt,h.isLt⟩ k active d.val d.isLt,
      WithBot.unbotD_coe]
  · apply Finset.sum_congr rfl
    intro k hk
    have active : k < x.counts b.val := Finset.mem_range.mp hk
    rw [host_logit c x b h k active,
      valid.logics ⟨b.val,h.val⟩ ⟨b.isLt,h.isLt⟩ k active]

theorem old_real_domain (c : Host.Call) (x : Host.Input (WithBot ℝ))
    (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (valid : NormalizableInputs c x logics values) : RealDomain (hostInputs c x) :=
  ⟨hostInputs_match c x,host_normalizable c x logics values valid⟩

/-- The public output window excludes exactly the old complete-grid exterior,
including zero dimensions and empty batch/head grids. -/
theorem frame_scope (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (work : List Host.Job) (xs : Inputs A c counts)
    (r : RegionName) (a : Nat) :
    (sourceIO M encode c observe counts work).Outside xs r a ↔ Host.Outside c 0 r a := by
  simp only [Host.Outside,Nat.not_lt_zero,false_or,Nat.sub_zero]
  constructor
  · intro outside
    by_cases hr : r="Out"
    · right
      intro j member hit
      obtain ⟨d,hd,same⟩ := hit
      have legal := (Host.mem_workset c j).mp member
      have off := outside.1 (0 : Fin 1) (⟨j.batch,legal.1⟩,⟨j.head,legal.2⟩,⟨d,hd⟩,()) trivial
      exact off.elim (fun no => no hr) (fun no => no same)
    · exact Or.inl hr
  · intro outside
    constructor
    · intro port i _
      have hp : port=(0 : Fin 1) := by
        apply Fin.ext
        change port.val=0
        have : port.val<1 := port.isLt
        omega
      subst port
      change r≠"Out" ∨ a≠Host.address c ⟨i.1.val,i.2.1.val⟩ i.2.2.1.val
      rcases outside with hr | off
      · exact Or.inl hr
      · right
        intro same
        exact off ⟨i.1.val,i.2.1.val⟩ ((Host.mem_workset c _).mpr ⟨i.1.isLt,i.2.1.isLt⟩)
          ⟨i.2.2.1.val,i.2.2.1.isLt,same⟩
    · intro k
      exact k.elim0

#axiomsClean native_host_target
#axiomsClean real_host_target
#axiomsClean old_real_domain
#axiomsClean frame_scope
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
