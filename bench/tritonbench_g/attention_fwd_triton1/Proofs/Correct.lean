import bench.tritonbench_g.attention_fwd_triton1.Proofs.Host

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

/-- Whole actual-precision invocation. O/H equal the concrete sourceFP word
graph; read-only and outside words retain their exact bits. Real mathematics
has its own independent theorem and is not a numerical premise here. -/
structure ResultCorrect (c : Shape) (abi : ABI c) (actualScale : BitVec 32)
    (store ifcond : Bool) (initial final : Memory (BitVec 32)) : Prop where
  output : ∀ head row j, head<c.heads → row<c.T → j<c.D →
    final (abi.o.base+c.mainOffset head row j)=
      SourceFP.output c (inputsOf c abi initial) actualScale head row j ifcond
  history : store=Bool.true → ∀ head chunk i j,
    head<c.heads → chunk<c.chunks → i<c.D → j<c.D →
    final (abi.h.base+c.historyOffset head chunk i j)=
      SourceFP.history c (inputsOf c abi initial) actualScale head chunk i j ifcond
  historyDisabled : store=Bool.false → ∀ point, abi.h.Contains point → final point=initial point
  readOnly : ReadOnly c abi final initial
  frame : ∀ point, ¬abi.o.Contains point → (store=Bool.false ∨ ¬abi.h.Contains point) →
    final point=initial point

theorem native_source_complete (native : ISA) (dom : NumericDomain)
    (c : Shape) (abi : ABI c) (actualScale : BitVec 32) (store ifcond : Bool)
    (initial : Memory (BitVec 32))
    (numeric : BitNumerical native dom c (inputsOf c abi initial) actualScale ifcond) :
    ResultCorrect c abi actualScale store ifcond initial
      (Source.run (wordOps native) c abi actualScale store ifcond initial) := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · exact FPEquiv.output native dom c abi actualScale store ifcond initial numeric
  · exact FPEquiv.history native dom c abi actualScale store ifcond initial numeric
  · exact Source.run_history_disabled (wordOps native) c abi actualScale store ifcond initial
  · exact Source.run_readOnly (wordOps native) c abi actualScale store ifcond initial
  · exact Source.run_frame (wordOps native) c abi actualScale store ifcond initial

/-- The complete concrete sourceFP word execution needs no numerical law. -/
theorem source_complete (c : Shape) (abi : ABI c) (actualScale : BitVec 32) (store ifcond : Bool)
    (initial : Memory (BitVec 32)) :
    ResultCorrect c abi actualScale store ifcond initial
      (Source.run (wordOps SourceFP.scalar) c abi actualScale store ifcond initial) :=
  native_source_complete SourceFP.scalar SourceFP.allOperands c abi actualScale store ifcond initial
    (reference_witness c (inputsOf c abi initial) actualScale ifcond)

structure PublicCorrect (c : Shape) (args : Host.Arguments) (receipt : Host.AllocationReceipt c args)
    (actualScale : BitVec 32) (store ifcond : Bool) (initial : Memory (BitVec 32)) (r : Host.Result) : Prop where
  publicOutput : Host.publicReturn r=Host.outputTensor c args receipt.abi
  historyDescriptor : r.history=Host.historyTensor c args receipt.abi
  launched : r.launched=decide (¬Host.Empty c)
  valuesAndFrame : ResultCorrect c receipt.abi actualScale store ifcond initial r.memory
  machine : Machine.Closed c receipt.abi

/-- Full returned word graph under two local scalar conformance families.
Host/compiler/runtime binding identifies the same actual scale word and
source instruction schedule. No exact FP-to-Real equality is assumed. -/
theorem native_complete (native : ISA) (dom : NumericDomain)
    (c : Shape) (args : Host.Arguments) (receipt : Host.AllocationReceipt c args)
    (actualScale : BitVec 32) (store ifcond : Bool)
    (backend : Host.AllocationReceipt c args → Host.Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32))
    (checks : Host.InputChecks c args) (hst : args.store=some store) (hic : args.ifcond=some ifcond)
    (machine : Machine.Domain c receipt.abi)
    (binding : Host.CompilerBinding native c args receipt actualScale store ifcond backend initial)
    (numeric : BitNumerical native dom c (inputsOf c receipt.abi initial) actualScale ifcond) :
    ∃ r, Host.run c args (some receipt) backend initial=some r ∧
      PublicCorrect c args receipt actualScale store ifcond initial r := by
  refine ⟨Host.result c args receipt.abi
    (Source.run (wordOps native) c receipt.abi actualScale store ifcond initial) (decide (¬Host.Empty c)),?_,?_⟩
  · exact Host.success native c args receipt actualScale store ifcond backend initial checks hst hic machine binding
  · refine ⟨rfl,rfl,rfl,?_,?_⟩
    · exact native_source_complete native dom c receipt.abi actualScale store ifcond initial numeric
    · exact Machine.closed c receipt.abi machine

#axiomsClean source_complete
#axiomsClean native_complete
#specNonCircular SourceFP.output avoiding [Source.run,ResultCorrect,native_complete]
#specNonCircular SourceFP.history avoiding [Source.run,ResultCorrect,native_complete]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
