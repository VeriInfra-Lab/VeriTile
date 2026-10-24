import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.ScheduleEquivalence

namespace VeriTile.Bench.Repairs.Fp4ToBf16.FPEquiv
open VeriTile Triton FP.Structural ScheduleIO
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 Host
open scoped VeriTile.Spec
noncomputable section

/-- The existing conversion candidate reorders the complete CTA schedule.
Both actual finite-ABI programs safely convert every packed byte and agree at
all BF16 outputs, with both exterior frames. The uint8 input domain, signed32
payload operations, FP32 bitcast, BF16 cast and both zero signs are retained. -/
specification fp4_to_bf16_schedule_equiv (h : Input)
    (storage : Finite.Storage h.shape.count) (a b : List Nat)
    (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (left : a.Perm (List.range (grid h))) (right : b.Perm (List.range (grid h))) :
    original h a ≡[assumptions] candidate h b :=
  ScheduleIO.equivalent h storage a b fresh gridSafe left right

#print_fp_assumptions fp4_to_bf16_schedule_equiv
#axiomsClean fp4_to_bf16_schedule_equiv
#stmtSurfaceSubset fp4_to_bf16_schedule_equiv ⊆
  [RegionName,Input,Input.input,Input.output,Input.shape,Shape.count,Finite.Storage,grid,
   original,candidate,assumptions,Spec.Equiv,ScheduleIO.Program,ScheduleIO.instProgramSyntaxProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.FPEquiv
