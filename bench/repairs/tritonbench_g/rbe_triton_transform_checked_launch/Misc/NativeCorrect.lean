import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.IOCorrectness
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelNative

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform Ordered
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The same FP32 source and full grid, interpreting native words through the
existing local contraction profile. All FP32 words remain admitted. -/
@[reducible] def nativeIO (p : Primitives) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) : ProgramIO Word Tagged :=
  rotationIO (operations p) r X Y qs leaf Tagged.word

/-- Independent ordered native rotation: trig-first operands and the selected
mul/FMA orders. These are primitive meanings, not transformation assumptions. -/
def reference (p : Primitives) (r : Request) (xs : TileIndex [r.size] → Word)
    (j : TileIndex [r.size]) : Word :=
  Ordered.result p r (logicalInput r xs)
    (coordinates r j.1.val).1 (coordinates r j.1.val).2.1 (coordinates r j.1.val).2.2

/-- Complete source-to-native observation with exact tagged output and frame.
No numerical rewrite or experimental premise is introduced. -/
theorem kernel_native_equivalence (p : Primitives) (r : Request)
    (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    (nativeIO p r X Y qs).Implements
      (fun xs => packOutput r Y (reference p r (xs 0))) := by
  have proved := IOCorrectness.implements (operations p) r X Y different qs all leaf Tagged.word
  have target_eq : IOCorrectness.target (operations p) r X Y qs leaf Tagged.word =
      (fun xs => packOutput r Y (reference p r (xs 0))) := by
    funext xs o
    have h : o=0 := Fin.eq_zero o
    subst o
    funext j
    exact source_profile p r (fun i => leaf (logicalInput r (xs 0) i)) _ _ _
  rw [target_eq] at proved
  exact proved

#specNonCircular reference avoiding [Selected.source,FP.Structural.exec,Runtime.run,scalarResult]
#axiomsClean kernel_native_equivalence
#stmtSurfaceSubset kernel_native_equivalence ⊆
  [Request,RegionName,Runtime.Program,Runtime.programs,Primitives,nativeIO,
   ProgramIO.Implements,packOutput,reference,Word,Tagged,instInhabitedTagged,
   ProgramIO.Inputs,IOChannel,MemoryIO.inputs,ProgramIO.toMemoryIO,
   List.instNeZeroNatLengthCons,IOChannel.mk,TileDType.fp32,Request.size]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.FPEquiv
