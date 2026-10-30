import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedContext
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

 theorem eval_nat (M : Algebra α) (p : Option ComputeDType) (n : Nat) (s : State α) :
    evalOp M p (.constNat n) s=some (fun _=>n) := by rw [evalOp_unfold]
 theorem eval_ref (M : Algebra α) (p : Option ComputeDType) (d : TileDType) (sh : TileShape) (n : RegName) (s : State α) :
    evalOp M p (.ref d sh n) s=s.regs d sh n := by rw [evalOp_unfold]
 theorem eval_add (M : Algebra α) (p : Option ComputeDType) {d : TileDType} (nd : NumericDType d)
    {a b out : TileShape} (bc : Broadcast a b out) (x : Op d a) (y : Op d b) (s : State α) :
    evalOp M p (.add nd bc x y) s=(do return bop (numeric M p .add nd) bc (←evalOp M p x s) (←evalOp M p y s)) := by rw [evalOp_unfold]
 theorem eval_mul (M : Algebra α) (p : Option ComputeDType) {d : TileDType} (nd : NumericDType d)
    {a b out : TileShape} (bc : Broadcast a b out) (x : Op d a) (y : Op d b) (s : State α) :
    evalOp M p (.mul nd bc x y) s=(do return bop (numeric M p .mul nd) bc (←evalOp M p x s) (←evalOp M p y s)) := by rw [evalOp_unfold]
 theorem eval_block (M : Algebra α) (p : Option ComputeDType) (r : RegionName) (base : Op .nat [])
    (parent block strides : List Nat) (offsets : List (Op .nat [])) (s : State α) :
    evalOp M p (.makeBlockPtrDynOffsets r base parent block strides offsets) s=
      (do let b ←evalOp M p base s
          let off ←evalOffsets M p offsets s
          return fun _=>⟨r,b (),parent,block,strides,off⟩) := by rw [evalOp_unfold]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
