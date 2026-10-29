import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Addresses
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.ABI

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
open VeriTile.Triton FP.Structural
noncomputable section

theorem check_input_iff (c : Call) (s n p : Nat → Int) :
    checkInput c s n p = .ok () ↔ InputContract c s n p := by
  have shape : InputContract c s n p ↔ MetadataContract c ∧
      (active c → ContentChecks c s n p) :=
    ⟨fun h => ⟨h.metadata,h.contents⟩,fun h => ⟨h.1,h.2⟩⟩
  rw [shape,←check_iff]
  unfold checkInput checkContents
  cases result : check c <;>
    by_cases a : active c <;> by_cases b : ContentChecks c s n p <;> simp_all

theorem check_call_arguments_iff (x v o p s n : Argument) (k : Maximum)
    (starts lengths pages : Nat → Int) :
    checkCallArguments x v o p s n k starts lengths pages = .ok () ↔
      AdmittedCallArguments x v o p s n k starts lengths pages := by
  cases x <;> cases v <;> cases o <;> cases p <;> cases s <;> cases n <;> cases k <;>
    simp [checkCallArguments,AdmittedCallArguments]
  split_ifs <;> simp_all [check_input_iff]

/-- Scalar signed metadata decoding; this contains no validity or bounds facts. -/
structure ContentDecoding (x : Host.Input α) (s n p : Nat → Int) : Prop where
  starts : ∀ b, x.starts b = (s b).toNat
  counts : ∀ b, x.counts b = (n b).toNat
  pages : ∀ a, x.pages a = (p a).toNat

theorem checked_content (c : Call) (s n p : Nat → Int)
    (h : ContentChecks c s n p) (x : Host.Input α) (decode : ContentDecoding x s n p) :
    Content c x := by
  constructor
  · intro b hb
    have bounds := h.1 b (List.mem_range.mpr hb)
    rw [decode.starts,decode.counts]
    omega
  · intro j hj k hk
    have count := decode.counts j.batch
    have bound := h.2 j.batch (List.mem_range.mpr hj.1) k
      (List.mem_range.mpr (by rw [←count];exact hk))
    rw [decode.pages]
    simp only [pageOffset,host,decode.counts]
    omega

def signedNativeInput (m : Host.Memory (BitVec 32)) : Host.Input (BitVec 32) :=
  Host.decodeMemory (fun w => w.toInt.toNat) m

def nativeObserved (m : Host.Memory (BitVec 32)) (region : RegionName) (i : Nat) : Int :=
  (m region i).toInt
def realObserved (m : Host.Memory (WithBot ℝ)) (region : RegionName) (i : Nat) : Int :=
  ⌊(m region i).unbotD 0⌋

theorem native_decoding (m : Host.Memory (BitVec 32)) :
    ContentDecoding (signedNativeInput m) (nativeObserved m "Starts")
      (nativeObserved m "Lengths") (nativeObserved m "Locations") :=
  ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩

theorem real_decoding (m : Host.Memory (WithBot ℝ)) :
    ContentDecoding (Host.decodeMemory Host.realIndex m) (realObserved m "Starts")
      (realObserved m "Lengths") (realObserved m "Locations") :=
  ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩

#axiomsClean check_input_iff
#axiomsClean check_call_arguments_iff
#axiomsClean checked_content
#axiomsClean native_decoding
#axiomsClean real_decoding
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
