import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.NativeInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.FPEquiv
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option autoImplicit false

/-- Full native observer of the actual FP32 quote, retaining the actual scale
word, every cast, disabled TF32, both branches and the complete grid. -/
def io := NativeIO.io
def formula := NativeIO.formula
def numerical := NativeIO.domain

/-- Source-to-native observation against the independent ordered FP32 scalar
recurrence. The only numerical premises are the unchanged local mulRN/fmaRN
conformance equations and their reached operand domains. This auxiliary source-to-native result has no floating rewrite assumption;
the independent full-program address optimization pair is in FPEquiv.lean. -/
theorem kernel_native (native : ISA) (dom : NumericDomain) (c : Shape)
    (actualScale : BitVec 32) (storeFlag ifcond : Bool) :
    (io native c actualScale storeFlag ifcond).ImplementsOn
      (numerical native dom c actualScale ifcond) (formula c actualScale ifcond) :=
  NativeIO.implements native dom c actualScale storeFlag ifcond

#stmtSurfaceSubset kernel_native ⊆
  [io,formula,numerical,ProgramIO.ImplementsOn,ISA,NumericDomain,Shape,Term,instInhabitedTerm,NativeIO.Word]
#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result]
#specNonCircular numerical avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.FPEquiv
