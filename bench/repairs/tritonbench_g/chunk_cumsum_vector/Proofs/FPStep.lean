import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep
open VeriTile.Triton FP.Structural IndexProgram FPProgram FPValues
noncomputable section
set_option maxHeartbeats 4000000
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

def beforeStore (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (feature row q : Nat)
    (initial : State α) (acc : Values α .real [BS]) (u : State α) : State α :=
  let u1 := u.setReg "start_t" .nat [] (fun _ => q*c.BT)
  let u2 := u1.setReg "lanes_t" .nat [c.BT] (lanes c.BT)
  let u3 := u2.setReg "active_t" .bool [c.BT] (axisMask h.T c.BT (q*c.BT))
  let u4 := u3.setReg "times" .nat [c.BT] (coordinate h.T c.BT (q*c.BT))
  let u5 := u4.setReg "active" .bool [c.BT,BS] (mask h c feature q)
  let u6 := u5.setReg "p_s" .ptr [c.BT,BS] (pointer h c feature row q h.input h.inputStrides)
  let u7 := u6.setReg "p_z" .ptr [c.BT,BS] (pointer h c feature row q h.output h.outputStrides)
  let u8 := u7.setReg "b_s" .real [c.BT,BS] (loaded M h c feature row initial q)
  u8.setReg "b_c" .real [c.BT,BS] (output M dot c (loaded M h c feature row initial q) acc)
def result (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (feature row q : Nat)
    (initial : State α) (acc : Values α .real [BS]) (u : State α) : State α :=
  (FPMemory.scatter M h c feature row q (output M dot c (loaded M h c feature row initial q) acc)
    (beforeStore M dot h c feature row q initial acc u)).setReg
    "b_z" .real [BS] (updated M c (loaded M h c feature row initial q) acc)

theorem executes (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (feature row q : Nat)
    (initial : State α) (acc : Values α .real [BS]) (u : State α)
    (ht : u.regs .nat [] "i_t"=some (fun _ => q))
    (hb : u.regs .nat [] "i_b"=some (fun _ => row/h.H))
    (hh : u.regs .nat [] "i_h"=some (fun _ => row%h.H))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS)))
    (hj : u.regs .nat [BS] "columns"=some (coordinate h.S BS (feature*BS)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c))
    (ha : u.regs .real [BS] "b_z"=some acc)
    (read : ∀ off, u.mem h.input off=initial.mem h.input off) (padding : u.undef=initial.undef) :
    run (machine M dot) (loopBody h c) u=some (result M dot h c feature row q initial acc u) := by
  unfold loopBody
  rw [assign_alg _ _ _ _ _ _ _ _ (start_eval (machine M dot) "i_t" c.BT q u ht)]
  rw [assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.arange c.BT) _=some (lanes c.BT) from rfl)]
  rw [assign_alg _ _ _ _ _ _ _ _ (active_eval (machine M dot) h.T c.BT (q*c.BT) "lanes_t" "start_t" _ (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (coordinate_eval (machine M dot) h.T c.BT (q*c.BT) "start_t" "active_t" "lanes_t" _ (by simp) (by simp) (by simp))]
  rw [assign_alg _ _ _ _ _ _ _ _ (mask_eval (machine M dot) h c feature q _ (by simp) (by simp [hs]))]
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M dot) h c feature row q h.input h.inputStrides _ (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj]))]
  rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval (machine M dot) h c feature row q h.output h.outputStrides _ (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj]))]
  erw [assign_compute _ _ _ _ _ _ _ (load_eval M dot h c feature row q initial _ (by simp) (by simp) (by simpa using read) (by simpa using padding))]
  erw [assign_compute _ _ _ _ _ _ _ (output_eval M dot c (loaded M h c feature row initial q) acc _ (by simp [ha]) (by simp [hm]) (by simp))]
  rw [run,FPMemory.store_eval M dot h c feature row q (output M dot c (loaded M h c feature row initial q) acc) _ (by simp) (by simp) (by simp)]
  change run (machine M dot) [.ifThen _ _]
    (FPMemory.scatter M h c feature row q (output M dot c (loaded M h c feature row initial q) acc)
      (beforeStore M dot h c feature row q initial acc u))=_
  have guard : FP.Structural.evalOp (machine M dot) none (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      (FPMemory.scatter M h c feature row q (output M dot c (loaded M h c feature row initial q) acc)
        (beforeStore M dot h c feature row q initial acc u))=some (fun _ => Bool.true) := by
    simp [FP.Structural.evalOp,FPMemory.scatter,beforeStore,State.masked_scatter_eq_filter,ht,natLe,bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
    funext idx
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rw [run,step,guard]
  try simp only [Option.bind_some,↓reduceIte]
  rw [assign_compute _ _ _ _ _ _ _ (carry_eval M dot c (loaded M h c feature row initial q) acc _
    (by simp [FPMemory.scatter,beforeStore,State.masked_scatter_eq_filter,ha])
    (by simp [FPMemory.scatter,beforeStore,State.masked_scatter_eq_filter]))]
  simp only [run,Option.bind_some]
  rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep
