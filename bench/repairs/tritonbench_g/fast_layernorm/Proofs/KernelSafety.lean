import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelAffine
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool)
    (u : State α) (bounds : RegionName → Nat)
    (hx : ∀ j : Fin B,j.val<N → u.pids 0*N+j.val<bounds "x")
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds "w")
    (hb : bias=Bool.true → ∀ j : Fin B,j.val<N → j.val<bounds "b")
    (hy : ∀ j : Fin B,j.val<N → u.pids 0*N+j.val<bounds "y")
    (hr : u.pids 0<bounds "r") (hm : u.pids 0<bounds "mu") :
    TraceSafe (engine M) bounds (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u := by
  cases bias
  all_goals
    simp [TraceSafe,forwardKernel,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
      FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      hx,hw,hb,hy,hr,hm]
  · exact ⟨hx,hw,hy⟩
  · exact ⟨hx,hw,hb rfl,hy⟩

theorem backward (M : Algebra α) (N B : Nat) (u : State α) (bounds : RegionName → Nat)
    (hx : ∀ j : Fin B,j.val<N → u.pids 0*N+j.val<bounds "x")
    (hdy : ∀ j : Fin B,j.val<N → u.pids 0*N+j.val<bounds "dy")
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds "w")
    (hdx : ∀ j : Fin B,j.val<N → u.pids 0*N+j.val<bounds "dx")
    (hr : u.pids 0<bounds "r") (hm : u.pids 0<bounds "mu") :
    TraceSafe (engine M) bounds (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u := by
  simp [TraceSafe,backwardKernel,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
    FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
    hx,hdy,hw,hdx,hr,hm]
  exact ⟨hdy,hx,hw,hdx⟩

open KernelAffine in
theorem affine_iteration (M : Algebra α) (rows N wc bc pid row : Nat) (bias : Bool)
    (u : State α) (a b : Values α .real [128]) (bounds : RegionName → Nat)
    (hc : u.regs .nat [128] "columns"=some (columns pid))
    (hm : u.regs .bool [128] "active"=some (mask N pid))
    (ha : u.regs .real [128] "dw"=some a) (hb : u.regs .real [128] "dbias"=some b)
    (hx : ∀ j : Fin 128,pid*128+j.val<N → row*N+(pid*128+j.val)<bounds "x")
    (hdy : ∀ j : Fin 128,pid*128+j.val<N → row*N+(pid*128+j.val)<bounds "dy")
    (hr : row<bounds "r") (hmu : row<bounds "mu") :
    traceSafeList (engine M) bounds (body rows N wc bc bias)
      (u.setReg "row" .nat [] (fun _ => row)) := by
  simp [body,kernel,affineKernel,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
    columns,mask,hc,hm,ha,hb,hr,hmu,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro j hj; simpa [hj] using hdy j hj
  · intro j hj; simpa [hj] using hx j hj

open KernelAffine in
theorem affine_range (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M rows N pid initial n u)
    (bounds : RegionName → Nat)
    (hx : ∀ row,row<rows → ∀ j : Fin 128,pid*128+j.val<N → row*N+(pid*128+j.val)<bounds "x")
    (hdy : ∀ row,row<rows → ∀ j : Fin 128,pid*128+j.val<N → row*N+(pid*128+j.val)<bounds "dy")
    (hr : ∀ row,row<rows → row<bounds "r") (hm : ∀ row,row<rows → row<bounds "mu") :
    rangeSafe (engine M) bounds "row" n rows 1 (body rows N wc bc bias) u := by
  rw [rangeSafe_unfold]
  simp only [one_ne_zero,if_false]
  by_cases hn : n<rows
  · rw [if_pos hn]
    refine ⟨affine_iteration M rows N wc bc pid n bias u _ _ bounds
      inv.col inv.mask inv.dw inv.db (hx n hn) (hdy n hn) (hr n hn) (hm n hn),?_⟩
    obtain ⟨v,he,hmem,hcol,hmask,ha,hb⟩ := iteration M rows N wc bc pid n bias u
      (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)
      inv.col inv.mask inv.dw inv.db
    rw [he]
    have changeCarry : next M N pid n u
        (fun i => (carry M N pid initial n i).1) (fun i => (carry M N pid initial n i).2)=
        carry M N pid initial (n+1) := by
      funext i
      simp [next,carry,KernelReference.affineCarry,raw,saved,inv.mem]
    have nextInv : Holds M rows N pid initial (n+1) v :=
      ⟨Nat.succ_le_of_lt hn,hcol,hmask,by rw [changeCarry] at ha; exact ha,
        by rw [changeCarry] at hb; exact hb,hmem.trans inv.mem⟩
    exact affine_range M rows N wc bc pid bias initial (n+1) v nextInv bounds hx hdy hr hm
  · simp [hn]
termination_by rows-n

open KernelAffine in
theorem affine_tail (M : Algebra α) (rows N wc bc pid : Nat) (bias : Bool)
    (initial u : State α) (inv : Holds M rows N pid initial rows u)
    (bounds : RegionName → Nat)
    (hw : ∀ j : Fin 128,pid*128+j.val<wc → pid*128+j.val<bounds "dw")
    (hb : bias=Bool.true → ∀ j : Fin 128,pid*128+j.val<bc → pid*128+j.val<bounds "db") :
    traceSafeList (engine M) bounds (tail rows N wc bc bias) u := by
  cases bias
  all_goals
    simp [tail,kernel,affineKernel,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,State.setReg,
      FP.Structural.store,inv.col,inv.dw,inv.db,columns,
      ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex]
  · exact hw
  · exact ⟨hw,hb rfl⟩

open KernelAffine in
theorem affine (M : Algebra α) (rows N wc bc : Nat) (bias : Bool)
    (u : State α) (bounds : RegionName → Nat)
    (hx : ∀ row,row<rows → ∀ j : Fin 128,u.pids 0*128+j.val<N → row*N+(u.pids 0*128+j.val)<bounds "x")
    (hdy : ∀ row,row<rows → ∀ j : Fin 128,u.pids 0*128+j.val<N → row*N+(u.pids 0*128+j.val)<bounds "dy")
    (hr : ∀ row,row<rows → row<bounds "r") (hm : ∀ row,row<rows → row<bounds "mu")
    (hw : ∀ j : Fin 128,u.pids 0*128+j.val<wc → u.pids 0*128+j.val<bounds "dw")
    (hb : bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → u.pids 0*128+j.val<bounds "db") :
    TraceSafe (engine M) bounds (kernel rows N wc bc bias) u := by
  unfold TraceSafe
  rw [source_body,List.append_assoc,traceSafeList_append]
  refine ⟨?_,?_⟩
  · simp [prefixCode,kernel,affineKernel,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,natLt,numericLt,State.setReg]
  · intro ready he
    rw [prefix_executes] at he
    cases he
    rw [traceSafeList_append]
    refine ⟨?_,?_⟩
    · rw [traceSafeList_unfold]
      refine ⟨?_,?_⟩
      · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_some, and_self, true_and]
        exact affine_range M rows N wc bc (u.pids 0) bias u 0 (start M N u)
          (start_holds M rows N u) bounds hx hdy hr hm
      · cases step (engine M) (.forRangeDyn "row" (.constNat 0) (.constNat rows) (.constNat 1)
          (body rows N wc bc bias)) (start M N u) <;> simp only [traceSafeList_nil]
    · intro loopState ran
      obtain ⟨v,he,inv⟩ := loop_executes M rows N wc bc (u.pids 0) bias u 0 (start M N u)
        (start_holds M rows N u)
      simp [run,step,evalOp_unfold,Option.bind,he] at ran
      cases ran
      exact affine_tail M rows N wc bc (u.pids 0) bias u loopState inv bounds hw hb

end
end VeriTile.Bench.Repairs.FastLayernorm.KernelSafety
