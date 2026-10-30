import bench.tritonbench_g.matrix_reduction.Proofs.GraphBase

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelFP
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

/-- The backend retains native cast and max behavior, including NaNs and zero
 ties. No value-domain premise or arithmetic identity is used for execution. -/
theorem successful {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (s : State α) (M N : Nat) (hN : 0 < N) :
    ∃ t, exec A (kernel d "x" "out" N 1 1 M N) s = some t := by
  cases d <;> simp [kernel, source16, source32, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, State.setReg,
    TileShape.axisDim, hN, BlockPtr.inBounds, TileShape.insertAxisIndex,
    TileShape.eraseAxis, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType]

/-- Independent primitive graph. Max retains its actual backend tree and NaN
 policy; FP16 promotion, FP32 loads and destination conversion stay explicit. -/
def formula {α : Type} (A : Algebra α) (d : Storage) (M N : Nat)
    (xs : Fin M → Fin N → α) (i : Fin M) : α :=
  let values : Values α .real [M,N] := fun k => match d with
    | .fp16 => A.cast (some .fp32) .fp16 .real (xs k.1 k.2.1)
    | .fp32 => A.fp32Load (xs k.1 k.2.1)
  let reduced := A.reduceMax (some .fp32) ⟨1, by simp⟩ Bool.false values (i,())
  match d with
  | .fp16 => A.cast none .real .fp16 reduced
  | .fp32 => reduced

def sourceValues {α : Type} [Inhabited α] (d : Storage) (M N : Nat) (s : State α) :
    Fin M → Fin N → α :=
  fun i j => toFloat (channel d) ((s.mem "x" (i.val*N+j.val)).read (channel d).toTileDType)

def written {α : Type} [Inhabited α] (A : Algebra α) (d : Storage) (M N : Nat) (s : State α) :=
  ((TileShape.allIndices [M]).foldl (fun t i => t.write "out" i.1.val
    (.mk (channel d).toTileDType (ofFloat (channel d)
      (formula A d M N (sourceValues d M N s) i.1)))) s).mem

theorem source_graph {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (s : State α) (M N : Nat) (hN : 0 < N) :
    ∃ t, exec A (kernel d "x" "out" N 1 1 M N) s = some t ∧
      t.mem = written A d M N s := by
  cases d <;> simp [kernel, source16, source32, ComputeKernel.surfaceBody, FP.Structural.exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, State.setReg,
    TileShape.axisDim, hN, BlockPtr.inBounds, TileShape.insertAxisIndex,
    TileShape.eraseAxis, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    written, formula, sourceValues, channel, fold_memory]

/-- Successful source execution and every observable memory cell for all
 input payloads and arbitrary floating primitives, including native NaN rules. -/
theorem native_graph {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (s : State α) (M N : Nat) (hN : 0 < N) :
    ∃ t, exec A (kernel d "x" "out" N 1 1 M N) s = some t ∧
      (∀ i : Fin M, t.mem "out" i.val = .mk (channel d).toTileDType
        (ofFloat (channel d) (formula A d M N (sourceValues d M N s) i))) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_graph A d s M N hN
  refine ⟨t,he,?_,?_⟩
  · intro i
    rw [hm]
    exact State.scatter_readback "out" (fun k : TileIndex [M] => k.1.val) _ s
      (i,()) (lane_injective M)
  · intro r o outside
    rw [hm]
    apply State.scatter_frame
    rcases outside with hr | ho
    · exact Or.inl hr
    · right
      intro i hi same
      have := i.1.isLt
      omega

#axiomsClean successful
#axiomsClean source_graph
#axiomsClean native_graph
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelFP
