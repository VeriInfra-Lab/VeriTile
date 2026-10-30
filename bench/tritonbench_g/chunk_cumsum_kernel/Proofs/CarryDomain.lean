import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.SourceSafety
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.CarryDomain
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Source precision and cast order, without source execution or memory. -/
def chunkValues {α : Type} (ops : FPRow.Ops α) (BT : Nat) (xs : Nat → α)
    (q : Nat) : Values α .real [BT] :=
  fun i => ops.scalar.cast (some .fp32) .fp32 .real (xs (q*BT+i.1.val))
def chunkSum {α : Type} (ops : FPRow.Ops α) (BT : Nat) (xs : Nat → α) (q : Nat) : α :=
  ops.scalar.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (chunkValues ops BT xs q) ()
def carry {α : Type} (ops : FPRow.Ops α) (BT : Nat) (xs : Nat → α) : Nat → α
  | 0 => ops.scalar.literal (some .fp32) .real 0
  | q+1 => ops.scalar.binary (some .fp32) .real .add (carry ops BT xs q) (chunkSum ops BT xs q)

/-- Only additions whose carry is consumed by a later chunk are guarded.
The final carry, final unspecified tail and unrelated FP words are unrestricted. -/
def Finite {α : Type} (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (T BT : Nat) (xs : Nat → α) : Prop :=
  ∀ q,q+1<(T+BT-1)/BT →
    D .finite (carry ops BT xs q) ∧ D .finite (chunkSum ops BT xs q)

variable {α : Type} [Inhabited α]

def Ready (ops : FPRow.Ops α) (input : RegionName) (T BT pid q : Nat)
    (xs : Nat → α) (s : State α) : Prop :=
  s.regs .nat [] "i_bh" = some (fun _ => pid) ∧
  s.regs .real [] "b_z" = some (fun _ => carry ops BT xs q) ∧
  ∀ j,q*BT≤j → j<T → (s.mem input (pid*T+j)).read .fp32 = xs j

private theorem input_values (ops : FPRow.Ops α) (input : RegionName)
    (T BT pid q : Nat) (xs : Nat → α) (s : State α)
    (ready : Ready ops input T BT pid q xs s) (hfull : (q+1)*BT≤T) :
    (fun i : TileIndex [BT] =>
      if q*BT+i.1.val<T then (s.mem input (pid*T+(q*BT+i.1.val))).read .fp32
      else s.undef .fp32 input (pid*T+(q*BT+i.1.val))) = fun i => xs (q*BT+i.1.val) := by
  funext i
  have active : q*BT+i.1.val<T := by have := i.1.isLt; rw [Nat.add_mul] at hfull; omega
  rw [if_pos active]
  exact ready.2.2 _ (by omega) active

private theorem loaded_values (ops : FPRow.Ops α) (input : RegionName)
    (T BT pid q : Nat) (xs : Nat → α) (s : State α)
    (ready : Ready ops input T BT pid q xs s) (hfull : (q+1)*BT≤T) :
    (fun i : TileIndex [BT] => ops.scalar.cast (some .fp32) .fp32 .real
      (if q*BT+i.1.val<T then (s.mem input (pid*T+(q*BT+i.1.val))).read .fp32
       else s.undef .fp32 input (pid*T+(q*BT+i.1.val)))) = chunkValues ops BT xs q := by
  funext i
  exact congrArg (ops.scalar.cast (some .fp32) .fp32 .real)
    (congrFun (input_values ops input T BT pid q xs s ready hfull) i)

/-- The complete body establishes the independent next-carry recurrence and
preserves all future inputs, including when the source and output alias. -/
theorem body_next (ops : FPRow.Ops α) (input output : RegionName) (T BT pid q : Nat)
    (xs : Nat → α) (s : State α) (ready : Ready ops input T BT pid q xs s)
    (hfull : (q+1)*BT≤T) :
    ∃ t,run (engine ops.algebra) (FPIndexBridge.parts input output T BT).body
      (s.setReg "i_t" .int [] (fun _ => (q:Int)))=some t ∧
      Ready ops input T BT pid (q+1) xs t := by
  have values := loaded_values ops input T BT pid q xs s ready hfull
  obtain ⟨hp,hz,reads⟩ := ready
  simp (config := {maxSteps := 300000}) [FPIndexBridge.parts,repairedKernel,
    ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,FPRow.Ops.algebra,ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,hp,hz,TileShape.indexToList,Ready,
    carry,chunkSum,chunkValues]
  rw [values]
  constructor
  · funext i
    cases i
    rfl
  · intro j hj hjT
    rw [State.masked_scatter_frame]
    · exact reads j (by rw [Nat.add_mul] at hj; omega) hjT
    · right
      intro i _ _
      have := i.1.isLt
      rw [Nat.add_mul] at hj
      omega

/-- The actual add site reads exactly the logical carry and full chunk sum. -/
theorem site_iff (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T BT pid q : Nat) (xs : Nat → α) (s : State α)
    (ready : Ready ops input T BT pid q xs s) (hfull : (q+1)*BT≤T)
    :
    (Rewrite.additionSite input output T BT).Holds ops.algebra D
      (s.setReg "i_t" .int [] (fun _ => (q:Int))) ↔
      D .finite (carry ops BT xs q) ∧ D .finite (chunkSum ops BT xs q) := by
  have values := loaded_values ops input T BT pid q xs s ready hfull
  obtain ⟨hp,hz,_⟩ := ready
  simp (config := {maxSteps := 300000}) [Rewrite.additionSite,Rewrite.parts,FPIndexBridge.parts,
    repairedKernel,Source.prefixBefore,ComputeKernel.surfaceBody,Source.intRange?,Site.Holds,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPRow.Ops.algebra,
    ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,hp,hz,TileShape.indexToList]
  rw [values]
  intro _
  constructor
  · intro h
    exact h ()
  · intro h i
    cases i
    exact h

theorem site_holds (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T BT pid q : Nat) (xs : Nat → α) (s : State α)
    (ready : Ready ops input T BT pid q xs s) (hfull : (q+1)*BT≤T)
    (finite : D .finite (carry ops BT xs q) ∧ D .finite (chunkSum ops BT xs q)) :
    (Rewrite.additionSite input output T BT).Holds ops.algebra D
      (s.setReg "i_t" .int [] (fun _ => (q:Int))) :=
  (site_iff ops D input output T BT pid q xs s ready hfull).mpr finite

private theorem prelude_ready (ops : FPRow.Ops α) (input output : RegionName)
    (T BT : Nat) (xs : Nat → α) (s first : State α)
    (reads : ∀ j,j<T → (s.mem input (s.pids 0*T+j)).read .fp32=xs j)
    (ran : run (engine ops.algebra) (FPIndexBridge.parts input output T BT).before s=some first) :
    Ready ops input T BT (s.pids 0) 0 xs first := by
  simp [FPIndexBridge.parts,repairedKernel,ComputeKernel.surfaceBody,Source.intRange?,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPRow.Ops.algebra] at ran
  subst first
  simpa [Ready,carry] using reads

private theorem reachable_ready (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (xs : Nat → α) (s first : State α)
    (reads : ∀ j,j<T → (s.mem input (s.pids 0*T+j)).read .fp32=xs j)
    (pre : run (engine ops.algebra) (FPIndexBridge.parts input output T cfg.block).before s=some first)
    {cur : Int} {u : State α}
    (reach : IntRangeReachable (engine ops.algebra) "i_t" (((T+cfg.block-1)/cfg.block : Nat) : Int) 1
      (FPIndexBridge.parts input output T cfg.block).body 0 first cur u) :
    ∃ q : Nat,cur=q ∧ (q<(T+cfg.block-1)/cfg.block → Ready ops input T cfg.block (s.pids 0) q xs u) := by
  induction reach with
  | initial => exact ⟨0,rfl,fun _ => prelude_ready ops input output T cfg.block xs s first reads pre⟩
  | @next cur a b reach nz active ran ih =>
    obtain ⟨q,rfl,ready⟩ := ih
    refine ⟨q+1,by simp,?_⟩
    intro later
    have fullChunk : (q+1)*cfg.block≤T :=
      Nat.le_of_lt (NativeIndex.chunk_start T cfg (q+1) later)
    obtain ⟨z,hz,hready⟩ := body_next ops input output T cfg.block (s.pids 0) q xs a
      (ready (by omega)) fullChunk
    have same : z=b := Option.some.inj (hz.symm.trans ran)
    simpa [same] using hready

/-- The logical domain implies exactly the original local reached-operand
condition. It assumes neither program success nor output correctness. -/
theorem of_inputs (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T : Nat) (cfg : Config) (xs : Nat → α) (s : State α)
    (reads : ∀ j,j<T → (s.mem input (s.pids 0*T+j)).read .fp32=xs j)
    (finite : Finite ops D T cfg.block xs) :
    Rewrite.FiniteCarry ops.algebra D input output T cfg.block s := by
  intro first pre q u reach later
  obtain ⟨k,hk,ready⟩ := reachable_ready ops input output T cfg xs s first reads pre reach
  have same : k=q := by omega
  subst k
  apply site_holds ops D input output T cfg.block (s.pids 0) q xs u (ready (by omega))
  · exact Nat.le_of_lt (NativeIndex.chunk_start T cfg (q+1) later)
  · exact finite q later

private theorem reachable_at (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (xs : Nat → α) (s first : State α)
    (reads : ∀ j,j<T → (s.mem input (s.pids 0*T+j)).read .fp32=xs j)
    (pre : run (engine ops.algebra) (FPIndexBridge.parts input output T cfg.block).before s=some first)
    (q : Nat) (active : q<(T+cfg.block-1)/cfg.block) :
    ∃ u,IntRangeReachable (engine ops.algebra) "i_t" (((T+cfg.block-1)/cfg.block : Nat) : Int) 1
      (FPIndexBridge.parts input output T cfg.block).body 0 first q u ∧
      Ready ops input T cfg.block (s.pids 0) q xs u := by
  induction q with
  | zero => exact ⟨first,.initial,prelude_ready ops input output T cfg.block xs s first reads pre⟩
  | succ q ih =>
    obtain ⟨u,reach,ready⟩ := ih (by omega)
    obtain ⟨v,ran,readyV⟩ := body_next ops input output T cfg.block (s.pids 0) q xs u ready
      (Nat.le_of_lt (NativeIndex.chunk_start T cfg (q+1) active))
    refine ⟨v,?_,readyV⟩
    simpa using IntRangeReachable.next reach (by decide : (1:Nat)≠0)
      (by exact_mod_cast (show q<(T+cfg.block-1)/cfg.block from by omega)) ran

/-- The logical numerical domain is equivalent to the previous reached-site
condition for the same bound inputs. No accepted numerical scope is narrowed. -/
theorem iff_inputs (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T : Nat) (cfg : Config) (xs : Nat → α) (s : State α)
    (reads : ∀ j,j<T → (s.mem input (s.pids 0*T+j)).read .fp32=xs j) :
    Rewrite.FiniteCarry ops.algebra D input output T cfg.block s ↔ Finite ops D T cfg.block xs := by
  refine ⟨?_,of_inputs ops D input output T cfg xs s reads⟩
  intro raw q later
  obtain ⟨first,pre,_,_⟩ := FPRow.prelude ops .fp32 .fp32 input output T cfg.block s
  have pre' : run (engine ops.algebra) (FPIndexBridge.parts input output T cfg.block).before s=some first := by
    rw [FPIndexBridge.before]
    exact pre
  obtain ⟨u,reach,ready⟩ := reachable_at ops input output T cfg xs s first reads pre' q (by omega)
  exact (site_iff ops D input output T cfg.block (s.pids 0) q xs u ready
    (Nat.le_of_lt (NativeIndex.chunk_start T cfg (q+1) later))).mp (raw first pre' q u reach later)

#axiomsClean iff_inputs

#axiomsClean body_next
#axiomsClean site_holds
#axiomsClean of_inputs

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.CarryDomain
