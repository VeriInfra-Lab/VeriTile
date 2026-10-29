/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.IO
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.FPEquiv
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- All original symbolic input terms retain identity encoding. Native
instruction/compiler observation is applied only to outputs. -/
abbrev decodeIO (g : PhiIO.Geometry) (order : List Nat) : ProgramIO Native.Term Native.Term :=
  PhiIO.decodeIO Native.algebra g order id id

def observer (p : KernelNative.Ops) (schedule : Native.Schedule) : Native.Term→Native.Word :=
  Native.eval (KernelNative.lower p) schedule

/-- Independent ordered scalar graph, retaining arbitrary symbolic operands,
all existing load boundaries and every original compiler fusion/order choice. -/
def target (p : KernelNative.Ops) (schedule : Native.Schedule) (g : PhiIO.Geometry)
    (xs : PhiIO.Inputs Native.Term g) :=
  PhiIO.result (g:=g) (fun row d => observer p schedule (PhiIO.nativeTerm g xs row.val d.val))

theorem kernel_native (p : KernelNative.Ops) (schedule : Native.Schedule)
    (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    (decodeIO g order).ObservesOn (fun _ _ => observer p schedule)
      (PhiIO.Fits g) (target p schedule g) := by
  have proved := ProgramIO.ObservesOn.of_implementsOn
    (NativeIO.implements g width order coverage) (fun _ _ => observer p schedule)
  have same : (fun xs => (decodeIO g order).mapOutputs (fun _ _ => observer p schedule)
      (PhiIO.result (fun row d => PhiIO.nativeTerm g xs row.val d.val)))=target p schedule g := by
    funext xs o
    refine Fin.cases ?_ (fun i => i.elim0) o
    rfl
  rw [same] at proved
  exact proved

#specNonCircular target avoiding [Grid.source,FP.Structural.exec,values,Native.algebra,decodeIO]
#stmtSurfaceSubset kernel_native ⊆
  [KernelNative.Ops,Native.Schedule,PhiIO.Geometry,PhiIO.Geometry.B,PhiIO.Geometry.H,
   PhiIO.Geometry.D,PhiIO.Geometry.BD,decodeIO,ProgramIO.ObservesOn,observer,PhiIO.Fits,target,
   Native.Term,Native.Word,Native.instInhabitedTerm,ProgramIO.mapOutputs._proof_1,
   IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape]
-- Ops and Schedule are retained primitive/compiler interpretation ports.
-- There is one source, no new floating rewrite law or experimental obligation.
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.FPEquiv
