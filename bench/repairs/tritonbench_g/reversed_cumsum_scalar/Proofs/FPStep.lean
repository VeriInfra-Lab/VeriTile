import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPStep
open VeriTile.Triton FP.Structural FPProgram FPValues
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem assign_alg (M : Algebra α) (d : TileDType) (sh : TileShape) (name : RegName)
    (e : Op d sh) (v : Values α d sh) (rest : List ComputeStmt) (u : State α)
    (he : FP.Structural.evalOp M none e u=some v) :
    run M (.assign d sh name (.alg e)::rest) u=run M rest (u.setReg name d sh v) := by
  simp [run,step,FP.Structural.evalExpr,he]

theorem assign_compute (M : Algebra α) (sh : TileShape) (name : RegName)
    (e : Op .real sh) (v : Values α .real sh) (rest : List ComputeStmt) (u : State α)
    (he : FP.Structural.evalOp M (some .fp32) e u=some v) :
    run M (.assign .real sh name (.compute (.alg .fp32 e))::rest) u=run M rest (u.setReg name .real sh v) := by
  simp only [run,step,FP.Structural.evalExpr,FP.Structural.evalComputeOp]
  change ((FP.Structural.evalOp M (some .fp32) e u).bind (fun value => some (u.setReg name .real sh value))).bind (run M rest)=_
  rw [he]
  rfl

def beforeStore (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (row q : Nat)
    (input : Nat → α) (acc : α) (u : State α) : State α :=
  let u0 := u.setReg "i_t" .int [] (fun _ => (q:Int))
  let u1 := u0.setReg "start_t" .int [] (fun _ => (q*c.BT:Int))
  let u2 := u1.setReg "lanes" .nat [c.BT] (lanes c)
  let u3 := u2.setReg "active" .bool [c.BT] (mask h c q)
  let u4 := u3.setReg "times" .int [c.BT] (times h c q)
  let u5 := u4.setReg "p_s" .ptr [c.BT] (pointer h c row q h.input h.inputStrides)
  let u6 := u5.setReg "p_o" .ptr [c.BT] (pointer h c row q h.output h.outputStrides)
  let u7 := u6.setReg "b_s" .real [c.BT] (loaded M h c input q)
  let u8 := u7.setReg "b_zz" .real [] (fun _ => total M h c input q)
  let u9 := u8.setReg "b_z" .real [] (fun _ => updated M h c input q acc)
  u9.setReg "b_o" .real [c.BT] (output M scan h c input q (updated M h c input q acc))

def result (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (row q : Nat)
    (input : Nat → α) (acc : α) (u : State α) : State α :=
  FPMemory.scatter M h c row q (output M scan h c input q (updated M h c input q acc))
    (beforeStore M scan h c row q input acc u)

theorem executes (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (row : Nat)
    (input : Nat → α) (acc : α) (j : Int) (u : State α) (hj : j<1)
    (start : (-j).toNat*c.BT<h.T)
    (hr : u.regs .int [] "__rev_i_t"=some (fun _ => j))
    (hh : u.regs .int [] "i_bh"=some (fun _ => (row:Int)))
    (ha : u.regs .real [] "b_z"=some (fun _ => acc))
    (read : ∀ t, t<h.T → toFloat h.inputDType
      ((u.mem h.input (h.inputStrides.address h.H row t)).read h.inputDType.toTileDType)=input t) :
    run (machine M scan) (loopBody h c) u=some (result M scan h c row (-j).toNat input acc u) := by
  unfold loopBody
  rw [assign_alg _ _ _ _ _ _ _ _ (index_eval (machine M scan) j u hj hr)]
  rw [assign_alg _ _ _ _ _ _ _ _ (FPValues.start_eval (machine M scan) c (-j).toNat _ (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M scan) none (.arange c.BT) _=some (lanes c) from rfl)]
  rw [assign_alg _ _ _ _ _ _ _ _ (active_eval (machine M scan) h c (-j).toNat _ start (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (time_eval (machine M scan) h c (-j).toNat _ (by simp) (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M scan) h c row (-j).toNat h.input h.inputStrides _ (by simp [hh]) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M scan) h c row (-j).toNat h.output h.outputStrides _ (by simp [hh]) (by simp))]
  erw [assign_compute _ _ _ _ _ _ _ (load_eval M scan h c row (-j).toNat input _ start (by simp) (by simp) (by intro t ht; simpa using read t ht))]
  erw [assign_compute _ [] _ _ _ _ _ (sum_eval M scan h c input (-j).toNat _ (by simp))]
  erw [assign_compute _ _ _ _ _ _ _ (carry_eval M scan h c input (-j).toNat acc _ (by simp [ha]) (by simp; rfl))]
  erw [assign_compute _ _ _ _ _ _ _ (output_eval M scan h c input (-j).toNat (updated M h c input (-j).toNat acc) _ (by simp) (by simp))]
  rw [run,FPMemory.store_eval M scan h c row (-j).toNat (output M scan h c input (-j).toNat (updated M h c input (-j).toNat acc)) _ (by simp) (by simp) (by simp)]
  simp only [Option.bind_some,run]
  rfl

#print axioms executes
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPStep
