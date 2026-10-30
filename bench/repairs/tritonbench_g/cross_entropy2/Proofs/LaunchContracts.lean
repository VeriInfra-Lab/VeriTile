import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardDispatch
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Native

namespace VeriTile.Bench.Repairs.CrossEntropy2.Native
open Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Native (Supported)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Ordinary CUDA launch ABI limits. Unsupported grids fail the local setup
operation before the source grid executes; this is not a caller-domain
restriction or a numerical assumption. Empty row batches skip setup. -/
structure GridLibrary {α σ : Type} (rt : Runtime α σ) : Prop where
  forward : ∀ p count before after, rt.setupForward2 p count before=.ok () after →
    p.rows≤2^31-1 ∧ count≤65535
  backward : ∀ p count before after, rt.setupBackward2 p count before=.ok () after →
    p.rows≤2^31-1 ∧ count≤65535

theorem forward_supported {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {x y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Return}
    (grid : GridLibrary rt) (e : ForwardData M rt o x y rows cols initial returned final) :
    rows=0 ∨ Supported rows e.count o.width cols := by
  by_cases empty : rows=0
  · exact Or.inl empty
  · have nonempty : e.parameters.rows≠0 := empty
    have call := e.launchCall
    simp only [forwardLaunch,parameters,if_neg empty] at call
    obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    cases u
    have limits := grid.forward e.parameters e.count _ _ setup
    rcases e.admitted with ⟨hc,hx,hy,hs,hd,hw,hl,hpow,hil,hiu,ht,htb,hclass,hsl,hsh,hz⟩
    exact Or.inr ⟨limits.1,limits.2,hw,hl,by omega⟩

theorem backward_supported {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
    (grid : GridLibrary rt) (e : BackwardData M rt o g x l y rows cols initial returned final) :
    rows=0 ∨ Supported rows e.count o.width cols := by
  by_cases empty : rows=0
  · exact Or.inl empty
  · have nonempty : e.parameters.rows≠0 := empty
    have call := e.launchCall
    simp only [backwardLaunch,parameters,if_neg empty] at call
    obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    cases u
    have limits := grid.backward e.parameters e.count _ _ setup
    rcases e.admitted with ⟨hc,hx,hy,hs,hd,hw,hl,hpow,hil,hiu,ht,htb,hclass,hsl,hsh,hz⟩
    exact Or.inr ⟨limits.1,limits.2,hw,hl,by omega⟩

#axiomsClean forward_supported
#axiomsClean backward_supported
#specNonCircular GridLibrary avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Native
