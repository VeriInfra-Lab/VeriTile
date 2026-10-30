import bench.repairs.tritonbench_g.rope_transform.Proofs.Real
import bench.repairs.tritonbench_g.rope_transform.Proofs.Physical

namespace VeriTile.Bench.Repairs.RopeTransform.Checked
open VeriTile Triton FP.Structural
noncomputable section

def invokeReal (r : Request) (call : Host.Preparation ℝ r) :=
  if check r call then Host.invoke Math.algebra r call else none

/-- All four returned descriptors, every caller-indexed mathematical value,
logical storage frame, and guard-derived finite-index/pointer limits. -/
def RealOutcome (r : Request) (call : Host.Preparation ℝ r) : Prop :=
  ∃ result,invokeReal r call=some result ∧
    (∀ key,result.tensor key=call.output key) ∧ result.cosine=call.cosine ∧ result.sine=call.sine ∧
    (∀ key b h s c,b<r.batches → h<r.heads key → s<r.sequence → c<r.dimension →
      (result.state.mem (result.tensor key).region ((result.tensor key).offset b h s c)).read .fp32=
        Math.expected r call.before (call.input key) key b h s c) ∧
    (∀ name o,Host.Outside r name o → result.state.mem name o=call.before.mem name o) ∧
    ABI.Limits r ∧ ABI.PointerLimits r call

theorem whole_real (r : Request) (call : Host.Preparation ℝ r) (h : check r call=Bool.true) :
    RealOutcome r call := by
  obtain ⟨result,run,objects,cosine,sine,values,frame⟩ := Math.whole_real r call
  exact ⟨result,by simp [invokeReal,h,run],objects,cosine,sine,values,frame,
    (launch_admitted r call h).1,(launch_admitted r call h).2⟩

/-- Complete guarded source execution, both returned tensor objects, every
FP32 output word and its four bytes, caller byte frame, and finite ABI bounds.
The coefficient return objects are included even on empty launches. -/
def FP32Outcome {M : Type} (ops : VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops M Native.Word Physical.Byte)
    (I : Native.Instructions) (S : Native.Schedule) (r : Request)
    (call : Host.Preparation Native.Term r) (memory : RegionName → Nat → Native.Word)
    (before prepared : M) : Prop :=
  ∃ result final,Physical.invoke ops I S r call prepared=some (result,final) ∧
    (∀ role,result.tensor role=call.output role) ∧ result.cosine=call.cosine ∧ result.sine=call.sine ∧
    (∀ role b h s c,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension →
      ops.read final (Physical.key (result.tensor role) ((result.tensor role).offset b h s c))=
        Native.expectedWord I S r memory (call.input role) role b h s c) ∧
    (∀ role b h s c n,b<r.batches → h<r.heads role → s<r.sequence → c<r.dimension → n<4 →
      ops.observe final (result.tensor role).device
        ((result.tensor role).base+4*((result.tensor role).offset b h s c)+n)=
        (Native.expectedWord I S r memory (call.input role) role b h s c).extractLsb' (8*n) 8) ∧
    (∀ owner address,Physical.Outside r call owner address →
      ops.observe final owner address=ops.observe before owner address) ∧
    ABI.Limits r ∧ ABI.PointerLimits r call

#axiomsClean whole_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeTransform.Checked
