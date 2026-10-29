import bench.repairs.tritonbench_g.attention_fwd_triton1.Contract

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked.FPEquiv
open Whole001
noncomputable section

/-- Whole native call with the original scale word, flags, tail masks,
pre-update history stores and ordered FP32 graph. Machine closure follows from
the actual guards. Only scalar mulRN/fmaRN bit conformance remains numerical. -/
specification checked_native (native : ISA) (dom : NumericDomain)
    (c : Shape) (args : Host.Arguments) (receipt : Host.AllocationReceipt c args)
    (pointers : Pointers receipt.abi) (actualScale : BitVec 32) (store ifcond : Bool)
    (backend : Host.AllocationReceipt c args → Host.Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32))
    (inputs : metadataChecks c args=Bool.true)
    (hst : args.store=some store) (hic : args.ifcond=some ifcond)
    (setup : setupChecks c pointers=Bool.true)
    (binding : Host.CompilerBinding native c args receipt actualScale store ifcond backend initial)
    (numeric : BitNumerical native dom c (inputsOf c receipt.abi initial) actualScale ifcond) :
    ∃ r, Host.run c args (some receipt) backend initial=some r ∧
      PublicCorrect c args receipt actualScale store ifcond initial r :=
  native_complete native dom c args receipt actualScale store ifcond backend initial
    ((metadataChecks_iff c args).1 inputs) hst hic
    (admitted_machine c receipt.abi pointers setup) binding numeric

#axiomsClean checked_native
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Checked.FPEquiv
