import bench.repairs.tritonbench_g.softmax_triton2.IO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1
open NativeGraph
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- Active native-source addresses come from the exact selected masks/pointers. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0<B)
    (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B,j.val<n → ib+s.pids 0*ir+j.val*ic<bounds inp)
    (output : ∀ j : Fin B,j.val<n → ob+s.pids 0*orow+j.val*oc<bounds out) :
    TraceSafe (Floating.backend M) bounds
      (SoftmaxTriton2.softmax_kernel out inp ib ob ir ic orow oc n B) s := by
  simp (maxSteps := 800000) [TraceSafe,SoftmaxTriton2.softmax_kernel,SoftmaxTriton1.softmax_kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,FP.Structural.State.setReg,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,FP.Structural.numericEq,ComputeDType.eraseDType,
    FP.Structural.ofFloat,FP.Structural.toFloat,TileShape.axisDim,hB,
    Region.cast,Option.bind,Floating.backend,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add] at input output ⊢
  exact ⟨input,output⟩

/-- Complete native source execution and typed owner-observed outputs, with
actual source access safety and exact raw exterior cells. -/
theorem implements (inst : Instructions) (out inp : RegionName)
    (rows cols ib ob ir ic orow oc B p : Nat) (hB : 0<B)
    (inj : ∀ i j : Fin B,i.val<cols → j.val<cols → i.val*oc=j.val*oc → i=j) :
    (io inst out inp rows cols ib ob ir ic orow oc B p).Implements
      (fun xs => pack out B (reference inst cols B (xs 0))) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial _
    apply (scheduleSafe_singleton ..).mpr
    apply source_safe (algebra inst 1) out inp ib ob ir ic orow oc cols B hB
    · intro j hj
      exact bounded.2.1 0 (j,PUnit.unit) hj
    · intro j hj
      exact bounded.2.2.1 0 (j,PUnit.unit) hj
  · intro xs initial bound
    let start := (launch inst out inp rows cols ib ob ir ic orow oc B p).initial initial
    obtain ⟨final,ran⟩ := Floating.row_exec (algebra inst 1) out inp ib ob ir ic orow oc cols B hB start
    refine ⟨final,?_,?_,?_⟩
    · exact (runSchedule_singleton ..).trans ran
    · intro o j hj
      have h : o=0 := Fin.eq_zero o
      subst o
      have value := Floating.row_value (algebra inst 1) out inp ib ob ir ic orow oc cols B hB inj
        start final ran j.1 hj
      have input : ∀ k : Fin B,k.val<cols →
          (start.mem inp (ib+start.pids 0*ir+k.val*ic)).read .real=atom (words B (xs 0) k.val) := by
        intro k hk
        have read := bound 0 (k,PUnit.unit) hk
        simpa [start,launch,Invocation.initial,bindRegisters,words,k.isLt,io,mapValue] using read
      rw [Floating.expression_congr (algebra inst 1) cols B _
        (fun k => atom (words B (xs 0) k.val)) input j.1] at value
      change mapCell (fun node : Node => node (owner 1 B j.1.val))
        (final.mem out (ob+p*orow+j.1.val*oc))=Cell.mk .real _
      change final.mem out (ob+p*orow+j.1.val*oc)=_ at value
      rw [value]
      simp only [mapCell,mapValue]
      apply congrArg (Cell.mk .real)
      simpa [Floating.formula,j.1.isLt,reference,pack] using
        source_graph_lowering inst 1 cols B (words B (xs 0)) j.1.val j.1.isLt
    · intro r a outside
      have framed := Floating.row_frame (algebra inst 1) out inp ib ob ir ic orow oc cols B hB
        start final ran r a
      apply framed
      intro j hj hit
      have miss := outside.1 0 (j,PUnit.unit) hj
      change r≠out ∨ a≠ob+p*orow+j.val*oc at miss
      exact miss.elim (fun h => h hit.1.symm) (fun h => h hit.2.symm)

#specNonCircular reference avoiding [SoftmaxTriton2.softmax_kernel,FP.Structural.exec,Floating.formula]
#axiomsClean source_safe
#axiomsClean implements
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.NativeIO
