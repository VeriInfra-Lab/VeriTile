import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.ScatterBridge
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Backends
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Output
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Real

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

theorem loop_model (M : Algebra α) (c : Shape) (s : State α) (scale : α) (plan : Plans c)
    (values : ∀ n (i : TileIndex [c.BV]),rowValues M c s scale n i=
      (ops M).storeCast (scalarPartial (ops M) c plan (inputs c (memory s)) scale
        (program s).z (program s).ik n (valueIndex c (program s) i.1.val)))
    (n : Nat) (r : FusedRwkv6Kernel.Region) (b : Nat) :
    (loopMemory M c s scale n (regionName r) b).read .real=
      (FusedRwkv6Kernel.runLoop (ops M) c plan (program s) scale (memory s) n).memory r b := by
  induction n generalizing r b with
  | zero => rfl
  | succ n ih =>
    rw [loopMemory,row_model M c s scale n plan _ (values n)]
    have hm : (fun r b => (loopMemory M c s scale n (regionName r) b).read .real)=
        (FusedRwkv6Kernel.runLoop (ops M) c plan (program s) scale (memory s) n).memory := by
      funext r b
      exact ih r b
    rw [hm]
    simp only [FusedRwkv6Kernel.runLoop,FusedRwkv6Kernel.step]
    have hv := funext fun l => FusedRwkv6Kernel.loop_partial (ops M) c plan (program s)
      (memory s) scale n l
    rw [hv]

theorem terminal_address (c : Shape) (s : State α) (i : TileIndex [c.BV,c.BK]) :
    terminalAddress c s i=stateAddress c (program s).z
      (keyIndex c (program s) i.2.1.val) (valueIndex c (program s) i.1.val) := by
  simp [terminalAddress,stateAddress,program,keyIndex,valueIndex,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc]

theorem terminal_hit (c : Shape) (s : State α) (r : FusedRwkv6Kernel.Region) (b : Nat)
    (hf : c.final=Bool.true) :
    (∃ i∈TileShape.allIndices [c.BV,c.BK],
      (valueMask c s i.1 && keyMask c s i.2.1)=Bool.true ∧ regionName r="ht" ∧
      b=terminalAddress c s i) ↔ FinalWrite c (program s) r b := by
  constructor
  · rintro ⟨i,_,ha,hr,hb⟩
    have ha' : valueIndex c (program s) i.1.val<c.V ∧ keyIndex c (program s) i.2.1.val<c.K := by
      simpa [valueMask,keyMask,valueIndex,keyIndex,program] using ha
    exact ⟨hf,(region_ht r).mp hr,i.1.val,i.1.isLt,i.2.1.val,i.2.1.isLt,
      ha'.1,ha'.2,hb.trans (terminal_address c s i)⟩
  · rintro ⟨_,hr,l,hl,j,hj,ha,hk,hb⟩
    let i : TileIndex [c.BV,c.BK] := (⟨l,hl⟩,⟨j,hj⟩,())
    refine ⟨i,TileShape.mem_allIndices _ i,?_,(region_ht r).mpr hr,?_⟩
    · simpa [valueMask,keyMask,valueIndex,keyIndex,program,i] using And.intro ha hk
    · exact hb.trans (terminal_address c s i).symm

/-- Full source-to-existing-model refinement. The row premise is discharged
below by the actual reduction backend, rather than exposed as a result oracle. -/
theorem program_model (M : Algebra α) (c : Shape) (s : State α) (scale : α) (plan : Plans c)
    (values : ∀ n (i : TileIndex [c.BV]),rowValues M c s scale n i=
      (ops M).storeCast (scalarPartial (ops M) c plan (inputs c (memory s)) scale
        (program s).z (program s).ik n (valueIndex c (program s) i.1.val)))
    (r : FusedRwkv6Kernel.Region) (b : Nat) :
    (finalMemory M c s scale (regionName r) b).read .real=
      FusedRwkv6Kernel.runProgram (ops M) c plan (program s) scale (memory s) r b := by
  unfold FusedRwkv6Kernel.runProgram
  rw [storeFinal_target (ops M) c plan (program s) (inputs c (memory s)) scale _ _
    (fun l j => congrFun (congrFun (loop_carry (ops M) c plan (program s) scale (memory s) c.T) l) j)]
  by_cases hf : c.final=Bool.true
  · let targetCell : Nat → Cell α := fun b => .mk .real
      (target (ops M) c plan (inputs c (memory s)) scale .ht b)
    have hv : ∀ i∈TileShape.allIndices [c.BV,c.BK],
        (valueMask c s i.1 && keyMask c s i.2.1)=Bool.true →
        terminalValue M c s i=targetCell (terminalAddress c s i) := by
      intro i _ hi
      have ha : valueIndex c (program s) i.1.val<c.V ∧ keyIndex c (program s) i.2.1.val<c.K := by
        simpa [valueMask,keyMask,valueIndex,keyIndex,program] using hi
      have dec:=state_decode c (program s).z (keyIndex c (program s) i.2.1.val)
        (valueIndex c (program s) i.1.val) ha.2 ha.1
      simp only [targetCell,target,terminal_address,dec.1,dec.2.1,dec.2.2]
      rfl
    unfold finalMemory
    rw [if_pos hf,fold_target _ _ "ht" _ _ targetCell hv]
    by_cases hw : FinalWrite c (program s) r b
    · rw [if_pos ((terminal_hit c s r b hf).mpr hw),if_pos hw]
      have hr:=hw.2.1
      subst r
      rfl
    · rw [if_neg (fun h => hw ((terminal_hit c s r b hf).mp h)),if_neg hw]
      exact loop_model M c s scale plan values c.T r b
  · have hw : ¬FinalWrite c (program s) r b := fun h => hf h.1
    simp only [finalMemory,hf,if_false,hw]
    exact loop_model M c s scale plan values c.T r b

theorem native_rows (p : Native.ReductionPlan) (c : Shape) (s : State Native.Term)
    (scale : Native.Term) (n : Nat) (i : TileIndex [c.BV]) :
    rowValues (Native.algebra p) c s scale n i=
      (ops (Native.algebra p)).storeCast (scalarPartial (ops (Native.algebra p)) c
        (Native.plans c p) (inputs c (memory s)) scale (program s).z (program s).ik n
          (valueIndex c (program s) i.1.val)) := by
  simp only [rowValues,Native.algebra,TileShape.axisDim,scalarPartial,Native.plans]
  apply congrArg Native.Term.store32
  apply tree_congr_on (a:=ops (Native.algebra p))
  intro j hj
  have h := legal_leaf (p.legal c.BK) hj
  simp [h,reductionInputs,terms,program,valueIndex,ops,Native.algebra,TileShape.insertAxisIndex]

theorem real_rows (c : Shape) (s : State ℝ) (scale : ℝ) (n : Nat) (i : TileIndex [c.BV]) :
    rowValues Real.algebra c s scale n i=
      (ops Real.algebra).storeCast (scalarPartial (ops Real.algebra) c
        (sequentialPlans c) (inputs c (memory s)) scale (program s).z (program s).ik n
          (valueIndex c (program s) i.1.val)) := by
  change (∑ j : Fin c.BK,scalarTerm realOps c (inputs c (memory s)) scale (s.pids 2) n
    (s.pids 1*c.BK+j.val) (s.pids 0*c.BV+i.1.val)) =
    (Tree.chain c.BK).eval 0 (·+·) (fun j => scalarTerm realOps c (inputs c (memory s)) scale
      (s.pids 2) n (s.pids 1*c.BK+j) (s.pids 0*c.BV+i.1.val))
  rw [tree_real _ _ (Tree.chain_legal c.BK)]
  exact Fin.sum_univ_eq_sum_range (fun j : Nat => scalarTerm realOps c (inputs c (memory s)) scale
    (s.pids 2) n (s.pids 1*c.BK+j) (s.pids 0*c.BV+i.1.val)) c.BK

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
