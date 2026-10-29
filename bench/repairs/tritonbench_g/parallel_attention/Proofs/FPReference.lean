import bench.repairs.tritonbench_g.parallel_attention.Spec
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostNative

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open HostAPI
set_option maxHeartbeats 1000000

theorem ordered_forward (a : Ops α) (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (entry : Memory α) (eps : α) (den dst : Nat) (both normalize : Bool)
    (fresh : normalize=true → both=false → HostNative.NormalizeLayout s output den dst)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let f:=HostNative.forward a c s x trees scratch output ks hs entry eps den dst both normalize
    Spec.Agrees s.T s.V (f.api.returned.read f.api.memory bh)
      (Spec.forward a c s trees (currentInputs a s (forwardBases x) entry) eps both normalize bh) := by
  have correct (kind : OutputKind) (hk : kind∈forwardKinds) (t e : Nat)
      (ht : t<s.T) (he : e<kind.width s) :
      compiledRun a c s (forwardBases x) trees scratch output ks hs entry
        (output kind+dense s.T (kind.width s) bh t e)=
      Spec.raw a c s trees (currentInputs a s (forwardBases x) entry) kind bh t e :=
    compiled_output a c s (forwardBases x) trees scratch output forwardKinds ks hs
      layout.kernelCoverage layout.hostCoverage layout.kernelFresh layout.kernelUnique
      layout.hostFresh layout.hostUnique entry kind hk _ (dense_bound _ _ _ _ _ _ hb ht he)
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) := correct .forward (by simp [forwardKinds]) t e ht he
  have hz (t : Nat) (ht : t<s.T) := correct .normalizer (by simp [forwardKinds]) t 0 ht (by norm_num [OutputKind.width])
  simp only [OutputKind.width] at ho hz
  have draw:=HostNative.dispatch_eq a c s (forwardBases x) trees scratch output forwardKinds ks hs layout entry
  cases both <;> cases normalize <;>
    simp only [HostNative.forward,HostNative.forwardMemory,returned,Returned.read,Spec.forward,Spec.Agrees,
      Bool.false_eq_true,if_false,if_true,draw]
  · intro t ht e he
    exact ho t ht e he
  · intro t ht e he
    let raw:=compiledRun a c s (forwardBases x) trees scratch output ks hs entry
    change HostNative.divisionRun a s (output .forward) den dst
      (HostNative.denominatorRun a eps s (output .normalizer) den raw) (dst+dense s.T s.V bh t e)=_
    rw [HostNative.division_output _ _ _ _ _ _ (by intro i hi;exact fun h=>(fresh rfl rfl).outputFresh i hi (Or.inl h))
      _ (dense_bound _ _ _ _ _ _ hb ht he)]
    rw [HostNative.denominator_frame _ _ _ _ _ _ _ (by
      intro i hi eq
      exact (fresh rfl rfl).denominatorFresh i hi (eq ▸ Or.inl
        ⟨Nat.le_add_right _ _,Nat.add_lt_add_left (dense_bound _ _ _ _ _ _ hb ht he) _⟩))]
    rw [dense_normalizer_index _ _ _ _ _ he,HostNative.denominator_output _ _ _ _ _ _
      (by intro i hi member;exact (fresh rfl rfl).denominatorFresh i hi (Or.inr member))
      _ (by simpa using dense_bound s.T 1 (s.B*s.H) bh t 0 hb ht (by omega))]
    change a.div (raw (output .forward+dense s.T s.V bh t e))
      (a.add (raw (output .normalizer+dense s.T 1 bh t 0)) eps)=_
    dsimp only [raw]
    rw [ho t ht e he,hz t ht]
  · exact ⟨ho,hz⟩
  · exact ⟨ho,hz⟩

#axiomsClean ordered_forward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelAttention
