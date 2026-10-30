import bench.tritonbench_g.layer_norm_liger.Proofs.WholeCheck
import bench.tritonbench_g.layer_norm_liger.Proofs.PublicReturn
import bench.tritonbench_g.layer_norm_liger.Proofs.NativeLaws
namespace VeriTile.Bench.Repairs.LayerNormLiger.PublicReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum softwareSum)
open Logical
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

theorem returned_logical (rest : Algebra ℝ) (fp : Forward.Parameters) (bp : Backward.Parameters)
    (s : State ℝ) (rows programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (h : Public.Layout bp programs dw db shape)
    (link : Public.Link fp bp rows) :
    ∃ f out, Grid.run (withSum (algebra rest) sumPrimitive) (Forward.kernel fp) s rows=some f ∧
      Host.forwardBackward (algebra rest) sumPrimitive Returned.scalar fp bp rows programs pw pb dw db shape s=some out ∧
      Public.Correct bp f programs dw db shape out ∧
      (∀ row, row<rows →
        Backward.mean bp (saved bp f) row=mean fp.columns (rowX fp (atPid s row)) ∧
        Backward.rstd bp (saved bp f) row=rstd fp.columns fp.epsilon (rowX fp (atPid s row))) ∧
      (∀ row, row<rows → ∀ i : TileIndex [fp.block], i.1.val<fp.columns →
        f.mem fp.y (row*fp.yStride+i.1.val)=.mk .fp32
          (y fp.columns fp.epsilon (rowX fp (atPid s row)) (weight fp s) (bias fp s) i.1.val)) :=
  Public.roundtrip rest fp bp s rows programs pw pb dw db shape h link

theorem returned_software_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt : FP.Value .fp32 → FP.Value .fp32)
    (fp : Forward.Parameters) (bp : Backward.Parameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State (FP.Value .fp32)) :
    Host.forwardBackward (Forward.software rest cfg literal rsqrt) (softwareSum cfg) (Host.scalarFP32 cfg)
      fp bp rows programs pw pb dw db shape s=
      some (Host.finish (Host.scalarFP32 cfg) bp programs pw pb dw db shape
        (Grid.reference (ClosedFP.backwardReference cfg literal bp)
          (Grid.reference (ClosedFP.forwardReference cfg literal rsqrt fp) s rows) programs)) :=
  Host.forwardBackward_bits rest cfg literal rsqrt fp bp rows programs pw pb dw db shape s

theorem returned_selected_plan_bits (P : NativePlan.Primitives)
    (contract : NativePlan.Contraction) (trees : NativePlan.Trees)
    (fp : Forward.Parameters) (bp : Backward.Parameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State NativePlan.Expr) :
    (Host.forwardBackward (NativePlan.algebra contract trees) (NativePlan.sums contract trees)
      (NativePlan.hostScalar contract) fp bp rows programs pw pb dw db shape s).map
        (fun out => fun r o => NativePlan.evalCell P (out.state.mem r o))=
      some (fun r o => NativePlan.evalCell P
        ((Host.finish (NativePlan.hostScalar contract) bp programs pw pb dw db shape
          (Grid.reference (Backward.reference (NativePlan.algebra contract trees) (NativePlan.sums contract trees) bp)
            (Grid.reference (Forward.reference (NativePlan.algebra contract trees) (NativePlan.sums contract trees) fp)
              s rows) programs)).state.mem r o)) :=
  NativePlan.complete_bits P contract trees fp bp rows programs pw pb dw db shape s

theorem selected_reduction_logical (input : FP.Value .fp32 → ℝ) (contract : NativePlan.Contraction)
    (pr : Option ComputeDType) {n : Nat} (p : ReductionPlan.Plan n) (xs : Fin n → NativePlan.Expr) :
    (p.tree.eval (.literal pr .fp32 0) (NativePlan.binary contract pr .fp32 .add) xs).ideal input=
      ((List.finRange n).map (fun i => (xs i).ideal input)).sum :=
  NativePlan.reduction_ideal input contract pr p xs

theorem selected_scalar_logical (input : FP.Value .fp32 → ℝ) (contract : NativePlan.Contraction)
    (pr : Option ComputeDType) (d : FloatDType) (op : Binary) (x y : NativePlan.Expr) :
    (NativePlan.binary contract pr d op x y).ideal input=
      NativePlan.realBinary op (x.ideal input) (y.ideal input) :=
  NativePlan.binary_ideal input contract pr d op x y

#axiomsClean returned_logical
#axiomsClean returned_software_bits
#axiomsClean returned_selected_plan_bits
#axiomsClean selected_reduction_logical
#axiomsClean selected_scalar_logical
#specNonCircular Public.Correct avoiding [Forward.kernel,Backward.kernel,Forward.reference,Backward.reference,
  Grid.reference,Host.backward,Host.forwardBackward]
#specNonCircular Host.axisSum avoiding [Logical.dx,Logical.dw,Logical.db,Forward.kernel,Backward.kernel]
#specNonCircular Host.finish avoiding [Logical.dx,Logical.dw,Logical.db,Forward.kernel,Backward.kernel]
#specNonCircular NativePlan.Expr.eval avoiding [Forward.kernel,Backward.kernel,Forward.reference,Backward.reference]
#specNonCircular NativePlan.binary avoiding [Forward.kernel,Backward.kernel,Forward.reference,Backward.reference]
#auditStmt returned_logical
#auditStmt returned_selected_plan_bits
end
end VeriTile.Bench.Repairs.LayerNormLiger.PublicReader
