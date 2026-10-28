import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.ScalarSource
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Output

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

/-- Lift a checked scalar-conversion step through the actual signed loop.
The local body obligation below is discharged by each complete raw body. -/
theorem scalar_loop_eq (M : Algebra α) (c : Shape) (initial : State α) (scale : α)
    (body : List ComputeStmt)
    (body_eq : ∀ (current : State α) (n : Nat) (ctx : Context M c initial scale n current),
      run (engine M) body (current.setReg "_" .int [] (fun _ => (n : Int)))=
        some (bodyExec M c initial current scale n ctx).val)
    (current : State α) (n : Nat) (hn : n≤c.T) (ctx : Context M c initial scale n current) :
    rangeInt (engine M) "_" n c.T 1 body current=
      rangeInt (engine M) "_" n c.T 1 (parts c).body current := by
  by_cases live : n<c.T
  · have live' : (n : Int)<(c.T : Int) := Int.ofNat_lt.mpr live
    rw [rangeInt,rangeInt]
    simp only [show (1 : Nat)≠0 by decide,if_false,if_pos live',
      body_eq current n ctx,(bodyExec M c initial current scale n ctx).property,Option.bind_some]
    simpa only [Nat.cast_add,Nat.cast_one] using
      scalar_loop_eq M c initial scale body body_eq
        (bodyExec M c initial current scale n ctx).val (n+1) (by omega)
        (body_context M c initial current scale n ctx live)
  · have live' : ¬(n : Int)<(c.T : Int) := by exact fun h => live (Int.ofNat_lt.mp h)
    rw [rangeInt,rangeInt]
    simp only [show (1 : Nat)≠0 by decide,if_false,if_neg live']
termination_by c.T-n
decreasing_by omega

private theorem scalar_source_eq (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (hs : s.regs .real [] "scale"=some (fun _ => scale))
    (code : ComputeKernel) (body : List ComputeStmt)
    (source : code.surfaceBody=(parts c).before++
      [.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1) body]++(parts c).after)
    (body_eq : ∀ (current : State α) (n : Nat) (ctx : Context M c s scale n current),
      run (engine M) body (current.setReg "_" .int [] (fun _ => (n : Int)))=
        some (bodyExec M c s current scale n ctx).val) :
    FP.Structural.exec (engine M) code s=FP.Structural.exec (engine M) (kernel c) s := by
  let start := entry M c s
  have loops := scalar_loop_eq M c s scale body body_eq start.val 0 (by omega)
    (entry_context M c s scale hs)
  change run (engine M) code.surfaceBody s=run (engine M) (kernel c).surfaceBody s
  rw [source,source_body,List.append_assoc,List.append_assoc,
    run_append,run_append,start.property]
  simpa only [Option.bind_some,List.singleton_append,run,FP.Structural.step,
    evalOp_unfold,evalOffsets] using
      congrArg (fun result => result.bind (run (engine M) (parts c).after)) loops


/-- Actual I32 source execution, including its implicit arithmetic conversion. -/
theorem i32_source_eq (M : Algebra α) (c : Shape) (s : State α) (value : Int)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromInt (some .fp32) value)) :
    FP.Structural.exec (engine M) (kernelI32 c value) s=
      FP.Structural.exec (engine M) (kernel c) s :=
  scalar_source_eq M c s (M.fromInt (some .fp32) value) hs
    (kernelI32 c value) (i32Parts c value).body (i32_body c value)
    (fun current n ctx => i32_body_exec M c s current value n ctx)

#axiomsClean i32_source_eq

/-- Actual I64 source execution, including its implicit arithmetic conversion. -/
theorem i64_source_eq (M : Algebra α) (c : Shape) (s : State α) (value : Int)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromInt (some .fp32) value)) :
    FP.Structural.exec (engine M) (kernelI64 c value) s=
      FP.Structural.exec (engine M) (kernel c) s :=
  scalar_source_eq M c s (M.fromInt (some .fp32) value) hs
    (kernelI64 c value) (i64Parts c value).body (i64_body c value)
    (fun current n ctx => i64_body_exec M c s current value n ctx)

#axiomsClean i64_source_eq

/-- Actual U64 source execution, including its implicit arithmetic conversion. -/
theorem u64_source_eq (M : Algebra α) (c : Shape) (s : State α) (value : Nat)
    (hs : s.regs .real [] "scale"=some (fun _ => M.fromNat (some .fp32) value)) :
    FP.Structural.exec (engine M) (kernelU64 c value) s=
      FP.Structural.exec (engine M) (kernel c) s :=
  scalar_source_eq M c s (M.fromNat (some .fp32) value) hs
    (kernelU64 c value) (u64Parts c value).body (u64_body c value)
    (fun current n ctx => u64_body_exec M c s current value n ctx)

#axiomsClean u64_source_eq
/-- Actual One source execution, including its implicit arithmetic conversion. -/
theorem one_source_eq (M : Algebra α) (c : Shape) (s : State α)
    (hs : s.regs .real [] "scale"=some (fun _ => M.literal (some .fp32) .real 1)) :
    FP.Structural.exec (engine M) (kernelOne c) s=
      FP.Structural.exec (engine M) (kernel c) s :=
  scalar_source_eq M c s (M.literal (some .fp32) .real 1) hs
    (kernelOne c) (oneParts c).body (one_body c)
    (fun current n ctx => one_body_exec M c s current n ctx)

#axiomsClean one_source_eq


end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
