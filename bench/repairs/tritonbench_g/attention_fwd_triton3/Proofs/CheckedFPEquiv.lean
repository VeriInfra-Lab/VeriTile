import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.SourceFPEquiv
import bench.repairs.tritonbench_g.attention_fwd_triton3.Contract
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
open VeriTile Triton Host Ownership Raw NativeSchedule
noncomputable section
theorem checked_native (c : Call) (A : FlatAlloc) (pointers : Pointers A)
    (args : Arguments) (_inputs : metadataChecks c args=Bool.true)
    (setup : setupChecks c A pointers=Bool.true) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (k : Kind) (profileOK : profileCheck c k=Bool.true) (choice : Selection c) (ops : Ops)
    (before : Physical.Bytes) (scale : Word) :
    Safety c A pointers ∧ (
    (∀ idx i b,active c i → FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale (Physical.address c A idx i b)=
      RawStores.byte (output c ops (schedule c k (admittedProfile c k profileOK) choice) (Physical.snapshot A before) scale idx i) b) ∧
    (∀ addr,(¬∃ idx i b,active c i ∧ Physical.address c A idx i b=addr) →
      FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale addr=before addr) ∧
    (Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale)).shape=
      [c.Z,c.H,if c.finish=0 then c.rounded else c.NQ,c.D] ∧
    (Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale)).sameM=Bool.true ∧
    (Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale)).sameL=Bool.true ∧
    (c.finish=0 → (Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice ops A before scale)).guaranteesSameO=Bool.true)) := by
  have safety:=admitted c A pointers setup
  refine ⟨safety,?_⟩
  exact FPEquiv.whole_bytes c safety.valid A safety.allocated safety.disjoint hc k
    (admittedProfile c k profileOK) choice ops before scale

/-- Same exact whole native bytes and caller-visible returns under only the
accepted nine local scalar/fixed-instruction law families. -/
theorem checked_equiv (c : Call) (A : FlatAlloc) (pointers : Pointers A)
    (args : Arguments) (_inputs : metadataChecks c args=Bool.true)
    (setup : setupChecks c A pointers=Bool.true)
    (k : Kind) (profileOK : profileCheck c k=Bool.true)
    (choice : Selection c) (actual expected : Ops) (laws : LocalLaws actual expected)
    (before : Physical.Bytes) (scale : Word) :
    Safety c A pointers ∧
    FPEquiv.reference c k (admittedProfile c k profileOK) choice actual A before scale=
      FPEquiv.reference c k (admittedProfile c k profileOK) choice expected A before scale ∧
    Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice actual A before scale)=
      Physical.returned c A (FPEquiv.reference c k (admittedProfile c k profileOK) choice expected A before scale) :=
  ⟨admitted c A pointers setup,
    FPEquiv.whole_equiv c A k (admittedProfile c k profileOK) choice actual expected laws before scale⟩

#axiomsClean checked_equiv

#axiomsClean checked_native
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
