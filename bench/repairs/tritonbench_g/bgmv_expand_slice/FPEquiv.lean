import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Physical

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPEquiv
open VeriTile.Triton Direct Direct.Native Contract
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

abbrev nativeReference := @Direct.Native.reference

/-- Complete actual typed source, independently expanded scalar-tree native
words, all four stored output bytes, and the full complement frame. Numerical
ports are fixed-arity scalar instructions; the compiler binding is structural. -/
theorem checked_fp (plans : Direct.FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    let h:=call x w y i s indices ok
    let result:=invoke plans S I binding x w y i s indices ok a before
    result.tensor=y ∧ result.returned=() ∧
    Direct.FPGrid.run scalarAlgebra (reduction plans) h (Direct.Geometry.programs h) (data a before)=
      some (Direct.FPGrid.fold scalarAlgebra (reduction plans) h (Direct.Geometry.programs h) (data a before)) ∧
    (∀ p : Direct.Geometry.Program h,indices p.batch≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (Direct.FPGrid.address h p)).address+byte.val)=
        Raw.unpack (nativeReference plans S I h a before p) byte) ∧
    (∀ owner address,(∀ p : Direct.Geometry.Program h,indices p.batch≠-1 →
        ¬(a.output.window.key (Direct.FPGrid.address h p)).covers owner address) → result.bytes owner address=before owner address) :=
  Direct.Native.whole plans S I binding x w y i s indices ok a before metadata

theorem checked_caller_bytes (plans : Direct.FPScalarTree.Plans) (S : NG.Selection) (I : NG.Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State)
    (owner address : Nat) (caller : a.input.backing.covers owner address ∨
      a.weights.backing.covers owner address ∨ a.indices.backing.covers owner address) :
    (invoke plans S I binding x w y i s indices ok a before).bytes owner address=before owner address :=
  Direct.Native.caller_frame plans S I binding x w y i s indices ok a before owner address caller

#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#axiomsClean Direct.Native.active_addresses
#axiomsClean Direct.Native.interrupted_frame
#axiomsClean Direct.Native.failure_frame
#axiomsClean Direct.Native.metadata_span
#axiomsClean Direct.Native.empty
#specNonCircular nativeReference avoiding [Direct.FPGrid.run,Direct.FPGrid.fold,Direct.Kernels.source,FP.Structural.exec,sourceStores]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.FPEquiv
