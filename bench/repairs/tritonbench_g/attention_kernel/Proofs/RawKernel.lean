import bench.repairs.tritonbench_g.attention_kernel.Proofs.Physical
import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeRouting

namespace VeriTile.Bench.Repairs.AttentionKernel.RawKernel
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 6000000

theorem no_grid (p : Parameters) (bm : 0<p.bm) (empty : ¬Live p) :
    programs (grid p)=[] := by
  by_cases hb : p.batches=0
  · simp [programs,grid,hb]
  by_cases hh : p.heads=0
  · simp [programs,grid,hh]
  have hn : p.length=0 := by unfold Live at empty;omega
  have divide : (p.length+p.bm-1)/p.bm=0 :=
    Nat.div_eq_of_lt (by omega)
  simp [programs,grid,divide]

theorem complete {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (p : Parameters) (hbm : 0<p.bm) (guard : Live p → Guard p) (separate : Separated p) (before : State α) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>dot) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,
        ∀ d (hd : d<p.dim),
        after.mem p.Out (logicalAddress p b h n d)=
          .mk p.storage.tile (ofFloat p.storage.float
            (answer M dot p
              (start (grid p) (n/p.bm,b*p.heads+h,0) before)
              (lane p hbm n d hd)))) ∧
      (∀ r off,Physical.Exterior p r off → after.mem r off=before.mem r off) := by
  by_cases live : Live p
  · have guard := guard live
    have unique : Unique p (grid p) work before := by
      intro a ha b hb i j hi hj same
      exact grid_unique _ guard before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
    obtain ⟨after,run,values,frame⟩ := run_correct (fun _=>M) (fun _=>dot) p
      separate guard.bn (grid p) work before unique
    refine ⟨after,run,?_,?_⟩
    · intro b hb h hh n hn d hd
      have active : Source.active p
          (start (grid p) (n/p.bm,b*p.heads+h,0) before)
          (lane p guard.bm n d hd) := by
        change n/p.bm*p.bm+n%p.bm < p.length
        simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
      have covers := (grid_covers p guard.bm b h n d hb hh hn hd).1
      have value := values _ (order.mem_iff.mp covers) _ active
      rw [physical_address _ guard.bm before b h n d hh hd] at value
      exact value
    · intro r off outside
      exact frame r off (Physical.exterior_outside _ before work order r off outside)
  · have nil : work=[] := List.nil_perm.mp (by simpa [no_grid p hbm live] using order)
    refine ⟨before,by simp [nil,runPrograms],?_,fun _ _ _=>rfl⟩
    intro b hb h hh n hn d hd
    exact False.elim (live ⟨by omega,by omega,by omega⟩)

/-- Decoded compiler routing without Tensor descriptors. -/
structure Plan (p : Parameters) (numWarps numStages : Nat) where
  config : NativeRouting.Configuration
  reflects : config.bm.value=p.bm ∧ config.bn.value=p.bn ∧ config.dim.value=p.dim ∧
    config.numWarps=numWarps ∧ config.stage.val+1=numStages ∧ config.storage=p.storage
  columns : NativePlan.WarpColumns
  columns_bound : columns.value≤numWarps
  routes : NativeRouting.Routes
  routing : NativeRouting.LocalRouting config.bn columns routes

def Plan.layout (plan : Plan p numWarps numStages) (scaleWord : BitVec 32) : FamilyNative.Layout :=
  ⟨plan.config.bn,plan.columns,scaleWord⟩

def Plan.ofTensor (plan : NativeRouting.CompilerPlan p abi) : Plan p abi.numWarps abi.numStages :=
  ⟨plan.config,plan.reflects,plan.columns,plan.columns_bound,plan.routes,plan.routing⟩

theorem Plan.layout_ofTensor (plan : NativeRouting.CompilerPlan p abi) (scaleWord : BitVec 32) :
    (Plan.ofTensor plan).layout scaleWord=plan.layout scaleWord := rfl

#axiomsClean complete
end
end VeriTile.Bench.Repairs.AttentionKernel.RawKernel

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
/-- Fixed scalar/conversion and m16n8k16 instruction coordinates. Raw-word
leaves are identity reads; no instruction can return a variable dot, reduction,
attention row or supplied final output. -/
structure Instructions where
  literal32 : ℝ → BitVec 32
  unary : Native.UnaryOp → BitVec 32 → BitVec 32
  binary : Native.BinaryOp → BitVec 32 → BitVec 32 → BitVec 32
  ternary : Native.TernaryOp → BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32
  mma16 : TypedSource.Storage → Fin 16 → Fin 8 →
    (Fin 16 → BitVec 32) → (Fin 16 → BitVec 32) → BitVec 32 → BitVec 32
def Instructions.isa (I : Instructions) : Native.ISA (BitVec 32) where
  bits := id
  literal32 := I.literal32
  unary := I.unary
  binary := I.binary
  ternary := I.ternary
  mma16 := I.mma16
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
