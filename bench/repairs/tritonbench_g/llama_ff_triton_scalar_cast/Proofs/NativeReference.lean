import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Logical

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeReference
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 6000000

/-- The four fixed K16 instructions of every source K64 block, flattened in
the selected instruction order. No result of the source execution occurs here. -/
def stream (count : Nat) := (List.range count).flatMap (fun t => (List.range 4).map (t,·))
def accumulation {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (r : RegionName) (row column count : Nat) :=
  (stream count).foldl (fun old e => a.mma16
    (fragment (rowScaled a p initial row e.1) e.2)
    (fragment (nativeRowW a p initial r column e.1) e.2) old) (a.literal (some .fp32) .real 0)
def moment {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (row count : Nat) (lane : Fin 64) :=
  (List.range count).foldl (fun old t => a.binary (some .fp32) .real .add old
    (a.powi (some .fp32) (a.cast (some .fp32) .fp16 .real (rowX a p initial row t lane)) 2))
      (a.literal (some .fp32) .real 0)
def normalization {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (row : Nat) :=
  a.unary (some .fp32) .rsqrt (a.binary (some .fp32) .real .add
    (a.binary (some .fp32) .real .div
      (LlamaFF.butterfly64.tree.eval (a.binary (some .fp32) .real .add)
        (moment a p initial row (steps p))) (a.fromInt (some .fp32) p.inner))
    (a.literal (some .fp32) .real p.epsilon))
def target {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial : State α) (row column : Nat) : α :=
  let rho := normalization a p initial row
  let w1 := accumulation a p initial "W1" row column (steps p)
  let w3 := accumulation a p initial "W3" row column (steps p)
  let mul := a.binary (some .fp32) .real .mul
  a.cast none .real .fp16 (mul (mul w3 rho) (mul (mul w1 rho)
    (g.sigmoidNeg (g.fma (g.neg w1) rho (a.literal (some .fp32) .real 0)))))

theorem accumulation_eq {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (r : RegionName) (row column count : Nat) :
    rowAcc a Bool.true p initial r row column count=accumulation a p initial r row column count := by
  induction count with
  | zero => rfl
  | succ t ih =>
    simp only [rowAcc,ite_true,ih,chain,show 64/16=4 by decide,accumulation,stream,
      List.range_succ,List.flatMap_append,List.flatMap_cons,List.flatMap_nil,List.append_nil,
      List.foldl_append,List.foldl_map]
theorem moment_eq {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (row count : Nat) (lane : Fin 64) :
    rowMoment a p initial row lane count=moment a p initial row count lane := by
  induction count with
  | zero => rfl
  | succ t ih => simp [rowMoment,ih,moment,List.range_succ,List.foldl_append]
theorem normalization_eq {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial : State α) (row : Nat) :
    rowNorm a p initial row=normalization a p initial row := by
  simp only [rowNorm,normalization,moment_eq]
  rfl
theorem logical_eq {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial : State α) (row column : Nat) :
    logicalNative a g p initial row column=target a g p initial row column := by
  simp only [logicalNative,target,accumulation_eq,normalization_eq]

#axiomsClean accumulation_eq
#axiomsClean moment_eq
#axiomsClean logical_eq
#specNonCircular target avoiding [source,run,nativeExecution,logicalNative,rowAcc,rowMoment]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.NativeReference
