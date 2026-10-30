import bench.tritonbench_g.rotary_emb.Proofs.TypedExecution

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Independent complete memory update with the actual Q-before-K order.
The second branch observes the current memory, so this lemma even preserves
aliases; the checked contract later discharges separation. -/
def reference {A : Type} [Inhabited A] (ops : Ops A) (p : Kernels.Layout)
    (s : State A) : State A :=
  let afterQ := branch ops p s Bool.false s
  branch ops p afterQ Bool.true afterQ

@[simp] theorem pair_projection (B : Nat) (i : Fin B) :
    (TileShape.dropInsertedIndex [1,B] (0 : Fin 3) 1 (0,0,i,PUnit.unit)).2.1=i := rfl

theorem int_double (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
theorem int_increment (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega

theorem execution_memory {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.Layout) (s : State A) :
    (execution rest ops p s).val.mem=(reference ops p s).mem := by
  simp only [execution,reference,branch,Pair.written,Pair.halfScatter,
    value,offset,coefficient,token,head,dimension,active,region,
    algebra,bop,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,
    masked_scatter_memory]
  dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  simp only [int_double,int_increment,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  simp (config := { maxSteps := 1000000 }) +contextual [and_assoc,pair_projection]
  try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  try simp only [int_double,int_increment,←Int.natCast_add,←Int.natCast_mul,
    Int.toNat_natCast,Int.ofNat_lt]
  try simp (config := { maxSteps := 1000000 }) +contextual [and_assoc]
  try rfl

#specNonCircular value avoiding [Kernels.kernel,sourcePair,execution]
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
