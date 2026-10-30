import bench.repairs.tritonbench_g.flash_decode2_phi.Contract
import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Host

namespace VeriTile.Bench.Repairs.FlashDecode2.Checked

theorem check_input_iff (c : Call) (lengths : Nat → Int) :
    checkInput c lengths = .ok () ↔ InputContract c lengths := by
  have contract : InputContract c lengths ↔ metadataChecks c ∧
      (active c → launchChecks c) ∧ (active c → contentChecks c lengths) :=
    ⟨fun h => ⟨h.metadata,h.launch,h.contents⟩,fun h => ⟨h.1,h.2.1,h.2.2⟩⟩
  rw [contract]
  unfold checkInput
  by_cases hm : metadataChecks c <;> by_cases ha : active c <;>
    by_cases hp : launchChecks c <;> by_cases hc : contentChecks c lengths <;>
    simp [hm,ha,hp,hc]

theorem check_arguments_iff (masked : Bool) (m e l o : Argument) (q : SequenceBlock)
    (lengths : Nat → Int) :
    checkArguments masked m e l o q lengths = .ok () ↔ AdmittedArguments masked m e l o q lengths := by
  cases m <;> cases e <;> cases l <;> cases o <;> cases q <;>
    simp only [checkArguments,AdmittedArguments] <;> try simp
  split <;> simp_all [check_input_iff]

def request (c : Call) (h : InputContract c lengths) : Host.Request :=
  ⟨geometry c,h.metadata.2.2.2.2.2.2.2.1⟩

#axiomsClean check_input_iff
#axiomsClean check_arguments_iff
end VeriTile.Bench.Repairs.FlashDecode2.Checked
