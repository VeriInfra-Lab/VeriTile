import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.ReaderFormat

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Independent normalized projection over exactly the logical K inputs. -/
def normalizedProjection (p : RmsMatmulRbe.Parameters ℝ) (x : RmsMatmulRbe.RowInput ℝ)
    (column : Nat) : ℝ :=
  (∑ k ∈ Finset.range p.K, x.x k*x.rms k*x.weight column k) *
    (1 / Real.sqrt ((∑ k ∈ Finset.range p.K, x.x k^2)/(p.K:ℝ)+p.epsilon))

/-- Q/K rotate adjacent projection pairs; V retains its projection. -/
def projectedRotary (p : RmsMatmulRbe.Parameters ℝ) (x : RmsMatmulRbe.RowInput ℝ)
    (rotary : Bool) (row column : Nat) : ℝ :=
  if rotary then
    let even := column-column%2
    let phase := ((p.start:ℝ)+(row:ℝ)) /
      Real.rpow p.theta (((even%p.headDim:Nat):ℝ)/(p.headDim:ℝ))
    if column%2=0 then
      normalizedProjection p x even*Real.cos phase - normalizedProjection p x (even+1)*Real.sin phase
    else
      normalizedProjection p x even*Real.sin phase + normalizedProjection p x (even+1)*Real.cos phase
  else normalizedProjection p x column

/-- Every scheduled Q→K→V implements the independent RMS/matmul/rotary
formula. Runtime EPS/THETA, original strides, both weight/output precision
branches and arbitrary output aliases are retained. The shared IO judgment
proves actual access safety, success, all final typed outputs and the frame.
Result channels observe the final winning writes in this finite schedule. -/
specification kernel_real (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : HostScalar) (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (order : List (Nat → Nat)) :
    Spec.Real (Logical.qkvIO fp8 p epsilon theta order ⊨{Logical.Domain fp8}
      fun xs phase launch rowLane colLane =>
        projectedRotary (parameters (p phase) epsilon.value theta.value)
          (Logical.rowInputs xs phase (Logical.batch order launch)
            (Logical.row p order phase launch rowLane))
          (decide (phase.val < 2)) (Logical.row p order phase launch rowLane)
          (Logical.column p order phase launch colLane)) := by
  have scalar : projectedRotary = realTarget := by
    funext param x rotary row column
    simp only [realTarget,RmsMatmulRbe.realTarget,RmsMatmulRbe.realProjection,
      RmsMatmulRbe.realNorm,HeadlineMath.rsum_eq,projectedRotary,normalizedProjection]
  rw [scalar]
  exact (ReaderFormat.statement_iff fp8 p epsilon theta order realTarget).mpr
    (IOResults.real_implements fp8 p epsilon theta order even apart)

#axiomsClean kernel_real
#stmtSurfaceSubset kernel_real ⊆
  [Arguments,RegionName,TypedSource.Arguments.N,TypedSource.Arguments.X,TypedSource.Arguments.W,
   TypedSource.Arguments.RMS,TypedSource.Arguments.OUT,HostScalar,HostScalar.value,
   Spec.Real,LogicalDomainIO.ImplementsOn,Logical.qkvIO,Logical.Domain,Logical.rowInputs,
   Logical.batch,Logical.row,Logical.column,Logical.Inputs,Logical.Outputs,parameters,
   IO.qkvIO,ArrayProgramIO.ImplementsOn,IO.inputs,IO.outputs,ArrayValues,ChannelValues,mathArithmetic,projectedRotary]
#specNonCircular projectedRotary avoiding [State,run,RawGrid.execute,rawTile,sourceTileValues,nativeTileValues]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Correct
