import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Misc.NativeIOResults
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.RawNative

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FPEquiv
open VeriTile Triton FP.Structural Whole KernelNative
noncomputable section

/-- Same full source schedule and IO, with address-labelled unrestricted FP32
words. Observation evaluates the established primitive graph and fixed scale. -/
def kernelIO (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (c : Call) (m : Metadata c) (work : List Program) :
    AddressedProgramIO Native.Word Native.Term :=
  Interface.io (algebras trees) c m (hostScale c.dim) work Native.Term.leaf
    (Native.eval I fusion (scaleWord c.dim))
abbrev formula := Reference.native

/-- Source-to-native observation of every score, with full raw typed frame.
Scalar add/mul/FMA meanings and shape-only reduction/contraction choices are
model parameters; this theorem introduces no floating transformation law. -/
theorem token_attention_kernel_fp_equiv (I : Native.Instructions)
    (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (c : Call) (m : Metadata c) (domain : Valid c m)
    (work : List Program) (order : (programs c.grid).Perm work) :
    (kernelIO I trees fusion c m work).ImplementsOn
      (Interface.MetadataValues c m) (formula I trees fusion c m) :=
  IOImplements.native I trees fusion c m domain work order

#stmtSurfaceSubset token_attention_kernel_fp_equiv ⊆
  [Native.Instructions,Native.Reductions,Native.Schedule,Native.Word,Native.Term,
   Native.instInhabitedTerm,Call,Metadata,Valid,Program,List.Perm,programs,Call.grid,
   AddressedProgramIO.ImplementsOn,kernelIO,Interface.MetadataValues,formula]
#specNonCircular formula avoiding [source,FP.Structural.exec,NativeGrid.runPrograms]
#axiomsClean token_attention_kernel_fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FPEquiv
