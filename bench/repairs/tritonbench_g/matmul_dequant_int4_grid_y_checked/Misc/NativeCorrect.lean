import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.ScalarIO

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.FPEquiv
open VeriTile.Triton FP.Structural Native
noncomputable section

/-- Exact selected-width native primitives; the 32-bit carrier holds FP32
words or zero-extended FP16 words. All original payloads remain admitted. -/
@[reducible] def nativeIO (f : Source.Profile) (I : Instructions) (p : Source.Parameters) :
    ProgramIO Word Word := PublicIO.dequantIO f (scalar I) p

/-- Independent selected-width signed conversion/multiplication formula. -/
def target (f : Source.Profile) (I : Instructions) (p : Source.Parameters)
    (xs : PublicIO.Inputs Word f p) :=
  PublicIO.result f p (fun x => match f with
    | .half => (I.mul16 (I.convert16 .signed (PublicIO.difference xs x))
        ((PublicIO.scale .half xs x).truncate 16)).zeroExtend 32
    | .single => I.mul32 (I.convert32 .signed (PublicIO.difference xs x)) (PublicIO.scale .single xs x))

/-- Full source-to-native observation with the existing finite half-conversion
port; no new rewrite law, input-value guard or compiler experiment is introduced. -/
theorem kernel_native (f : Source.Profile) (I : Instructions)
    (conversion : f=.half → Conversion I) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) :
    (nativeIO f I p).Implements (target f I p) := by
  have proved := PublicIO.implements f (scalar I) p valid
  have same : (fun xs => PublicIO.result f p (PublicIO.scalarFormula f (scalar I) p xs))=
      target f I p := by
    funext xs
    apply congrArg (PublicIO.result f p)
    funext x
    cases f <;> exact PublicIO.scalar_native _ I conversion p xs x
  rw [same] at proved
  exact proved

#specNonCircular target avoiding [Source.source,Source.execution,KernelGrid.result,nativeIO,scalar]
#stmtSurfaceSubset kernel_native ⊆
  [Source.Profile,Source.Profile.half,Source.Parameters,KernelGrid.Valid,Instructions,Conversion,
   nativeIO,ProgramIO.Implements,target,Word]
-- Instructions and finite Conversion retain the original primitive model ports.
-- This endpoint observes one source and does not assert an FP optimization law.
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.FPEquiv
