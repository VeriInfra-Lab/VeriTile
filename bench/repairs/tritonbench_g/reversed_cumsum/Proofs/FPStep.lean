import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPMemory
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPStep

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPStep
open VeriTile.Triton FP.Structural FPProgram FPIndices FPValues IndexProgram
open ReversedCumsumScalar.FPStep (assign_alg assign_compute)
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def indexed (h : Call) (c : Config) (q f : Nat) (u : State α) : State α :=
  let u0 := u.setReg "i_t" .int [] (fun _ => (q:Int))
  let u1 := u0.setReg "start_t" .int [] (fun _ => (q*c.BT:Int))
  let u2 := u1.setReg "start_s" .nat [] (fun _ => (f*BS))
  let u3 := u2.setReg "lane_t" .nat [c.BT] (lanes c.BT)
  let u4 := u3.setReg "lane_s" .nat [BS] (lanes BS)
  let u5 := u4.setReg "active_t" .bool [c.BT] (maskT h c q)
  let u6 := u5.setReg "active_s" .bool [BS] (maskS h f)
  let u7 := u6.setReg "times" .int [c.BT] (times h c q)
  let u8 := u7.setReg "features" .nat [BS] (features h f)
  u8.setReg "active" .bool [c.BT,BS] (mask h c q f)

def beforeStore (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (row q f : Nat)
    (input : Nat → Nat → α) (acc : Values α .real [BS]) (u : State α) : State α :=
  let u1 := (indexed h c q f u).setReg "p_s" .ptr [c.BT,BS] (pointer h c row q f h.input)
  let u2 := u1.setReg "p_z" .ptr [c.BT,BS] (pointer h c row q f h.output)
  let u3 := u2.setReg "b_s" .real [c.BT,BS] (loaded M h c f input q)
  u3.setReg "b_c" .real [c.BT,BS] (output M dot h c f input q acc)

def result (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (row q f : Nat)
    (input : Nat → Nat → α) (acc : Values α .real [BS]) (u : State α) : State α :=
  (FPMemory.scatter M h c row q f (output M dot h c f input q acc)
    (beforeStore M dot h c row q f input acc u)).setReg "b_z" .real [BS] (updated M h c f input q acc)

theorem executes (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (input : Nat → Nat → α) (acc : Values α .real [BS])
    (j : Int) (u : State α) (hj : j<1) (start : (-j).toNat*c.BT<h.T)
    (hr : u.regs .int [] "__rev_i_t"=some (fun _ => j))
    (his : u.regs .nat [] "i_s"=some (fun _ => f.val))
    (hih : u.regs .nat [] "i_bh"=some (fun _ => row))
    (htri : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c))
    (hacc : u.regs .real [BS] "b_z"=some acc)
    (read : ∀ t s, t<h.T → s<h.S → toFloat h.inputDType
      ((u.mem h.input (Geometry.address h row t s)).read h.inputDType.toTileDType)=input t s) :
    run (machine M dot) (loopBody h c) u=some (result M dot h c row (-j).toNat f.val input acc u) := by
  unfold loopBody indexOp
  rw [assign_alg _ _ _ _ _ _ _ _ (ReversedCumsumScalar.FPValues.index_eval (machine M dot) j u hj hr)]
  rw [assign_alg _ _ _ _ _ _ _ _ (startT_eval (machine M dot) c (-j).toNat _ (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (startS_eval (machine M dot) f.val _ (by simp [his]))]
  rw [assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.arange c.BT) _=some (lanes c.BT) from rfl)]
  rw [assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.arange BS) _=some (lanes BS) from rfl)]
  rw [assign_alg _ _ _ _ _ _ _ _ (activeT_eval (machine M dot) h c (-j).toNat _ start (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (activeS_eval (machine M dot) h f.val _ (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (time_eval (machine M dot) h c (-j).toNat _ (by simp) (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (feature_eval (machine M dot) h f.val _ (by simp) (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (mask_eval (machine M dot) h c (-j).toNat f.val _ (by simp) (by simp))]
  change run _ _ (indexed h c (-j).toNat f.val u)=_
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M dot) h c row (-j).toNat f.val h.input _
    (by simp [indexed,hih]) (by simp [indexed]) (by simp [indexed]))]
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M dot) h c row (-j).toNat f.val h.output _
    (by simp [indexed,hih]) (by simp [indexed]) (by simp [indexed]))]
  erw [assign_compute _ _ _ _ _ _ _ (load_eval M dot h c row (-j).toNat f input _ start
    (by simp) (by simp [indexed]) (by intro t s ht hs; simpa [indexed] using read t s ht hs))]
  erw [assign_compute _ _ _ _ _ _ _ (output_eval M dot h c f.val input (-j).toNat acc _
    (by simp [indexed,hacc]) (by simp [indexed,htri]) (by simp))]
  rw [run,FPMemory.store_eval M dot h c row (-j).toNat f.val (output M dot h c f.val input (-j).toNat acc) _
    (by simp) (by simp [indexed]) (by simp)]
  change run _ [.ifThen _ _] (FPMemory.scatter M h c row (-j).toNat f.val
    (output M dot h c f.val input (-j).toNat acc) (beforeStore M dot h c row (-j).toNat f.val input acc u))=_
  have hc : FP.Structural.evalOp (machine M dot) none nonnegativeOp
      (FPMemory.scatter M h c row (-j).toNat f.val (output M dot h c f.val input (-j).toNat acc)
        (beforeStore M dot h c row (-j).toNat f.val input acc u))=some (fun _ => Bool.true) := by
    simp [nonnegativeOp,FP.Structural.evalOp,FPMemory.scatter,beforeStore,indexed,
      State.masked_scatter_eq_filter,bop,numericLe,Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    change decide ((0:Int)≤max (-j) 0)=Bool.true
    simp
  rw [run,step,hc]
  rw [assign_compute _ _ _ _ _ _ _ (carry_eval M dot h c f.val input (-j).toNat acc _
    (by simp [FPMemory.scatter,beforeStore,indexed,State.masked_scatter_eq_filter,hacc])
    (by simp [FPMemory.scatter,beforeStore,State.masked_scatter_eq_filter]))]
  simp only [run,Option.bind_some]
  rfl

#print axioms executes
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPStep
