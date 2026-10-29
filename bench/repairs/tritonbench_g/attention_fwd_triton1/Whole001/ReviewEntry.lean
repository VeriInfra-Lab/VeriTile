import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Correct

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.ReviewEntry
noncomputable section

/-- A full public application: every ordinary and numerical premise is
explicit, with no postulated aggregate mathematical answer. -/
theorem full_public_application (isa : ISA) (dom : NumericDomain)
    (c : Shape) (args : Host.Arguments) (receipt : Host.AllocationReceipt c args)
    (scale : BitVec 32) (store ifcond : Bool)
    (backend : Host.AllocationReceipt c args → Host.Plan → Memory (BitVec 32) → Memory (BitVec 32))
    (initial : Memory (BitVec 32))
    (checks : Host.InputChecks c args) (hst : args.store=some store) (hic : args.ifcond=some ifcond)
    (machine : Machine.Domain c receipt.abi)
    (binding : Host.CompilerBinding isa c args receipt scale store ifcond backend initial)
    (laws : SourceFP.LocalConformance isa dom)
    (domain : GraphDomain isa dom c (inputsOf c receipt.abi initial) scale ifcond) :
    ∃ r, Host.run c args (some receipt) backend initial=some r ∧
      PublicCorrect c args receipt scale store ifcond initial r :=
  native_complete isa dom c args receipt scale store ifcond backend initial
    checks hst hic machine binding ⟨laws,domain⟩

#check TypedSource.repaired
#check Source.run
#check MathTarget.output
#check MathTarget.history
#check Host.plan
#check Host.CompilerBinding
#check GraphDomain
#check SourceFP.scalar
#check ResultCorrect
#check PublicCorrect

#axiomsClean Shape.padded_cover
#axiomsClean Shape.main_injective
#axiomsClean Source.step_readOnly
#axiomsClean Source.loop_carry
#axiomsClean Source.run_output
#axiomsClean Source.run_history
#axiomsClean Source.run_history_disabled
#axiomsClean Source.run_readOnly
#axiomsClean Source.run_frame
#axiomsClean RealProof.full_output
#axiomsClean RealProof.full_history
#axiomsClean SourceFP.transport
#axiomsClean reference_witness
#axiomsClean Graph.eval_output
#axiomsClean Graph.real_output
#axiomsClean Graph.eval_carry
#axiomsClean Graph.real_carry
#axiomsClean Machine.padded_coordinates
#axiomsClean Machine.pointer_exact
#axiomsClean Machine.closed
#axiomsClean Host.success
#axiomsClean FPEquiv.output
#axiomsClean FPEquiv.history
#axiomsClean source_complete
#axiomsClean native_complete
#axiomsClean full_public_application

#specNonCircular MathTarget.output avoiding [Source.run,Reference.output,native_complete]
#specNonCircular MathTarget.history avoiding [Source.run,Reference.carry,native_complete]
#specNonCircular SourceFP.output avoiding [Source.run,Source.step,native_complete]
#specNonCircular SourceFP.history avoiding [Source.run,Source.step,native_complete]
#specNonCircular SourceFP.LocalConformance avoiding [Graph.output,Graph.carry,Source.run,MathTarget.output]
#specNonCircular Host.CompilerBinding avoiding [MathTarget.output,MathTarget.history,Expr.real]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.ReviewEntry
