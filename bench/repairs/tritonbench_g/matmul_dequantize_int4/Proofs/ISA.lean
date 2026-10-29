import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPFusedBits
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Packing
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.ISA
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
open FPTyped
open VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
/-- Exactly one fixed K=16 tensor instruction result lane. The instruction
family is fixed by each compiled autotune profile, outside this scalar port. -/
abbrev MMA16 := (Fin 16 → Word) → (Fin 16 → Word) → Word → Word

def fragment {k : Nat} (x : Fin k → Word) (j : Nat) (e : Fin 16) : Word :=
  if h : j*16+e.val<k then x ⟨j*16+e.val,h⟩ else FP.Value.zero .fp32

def chain {k : Nat} (mma : MMA16) (a b : Fin k → Word) (acc : Word) : Word :=
  (List.range (k/16)).foldl (fun z j => mma (fragment a j) (fragment b j) z) acc

def tensor (mma : MMA16) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → Word)
    (b : TileIndex (batch ++ [k,n]) → Word)
    (c : TileIndex (batch ++ [m,n]) → Word) : TileIndex (batch ++ [m,n]) → Word :=
  match batch with
  | [] => fun i => chain mma (fun t => a (i.1,t,PUnit.unit))
      (fun t => b (t,i.2.1,PUnit.unit)) (c i)
  | _::tail => fun i => tensor mma (batch := tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length

def dot (mma : MMA16) : FPCompiled.DotAcc Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b c => tensor mma a b c⟩

/-- Initial-memory fixed-instruction recurrence. Every source K block carries
its live FP32 accumulator into the next; the final destination conversion is half. -/
def accum (mma : MMA16) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step+1 => fun i => chain mma
      (fun t => FPSoftware.machineA p s step (i.1,t,PUnit.unit))
      (fun t => FPFused.fusedWeight p s step (t,i.2.1,PUnit.unit)) (accum mma p s step i)
def output (mma : MMA16) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  VeriTile.Bench.TritonBenchG.QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (accum mma p s (steps p) i)

theorem accumulation_isa (mma : MMA16) (rest : Algebra Word)
    (p : Parameters) (s : State Word) (k : Nat) :
    FPFused.accumulation (FPSoftware.scalar rest) (dot mma) p s k=accum mma p s k := by
  induction k with
  | zero => simp [FPFused.accumulation,accum,FPSoftware.scalar,
      VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware.scalar,
      FP.Software.scalarFP32,FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [FPFused.accumulation,ih,FPFused.bValue_scalar,FPSoftware.aValue_scalar]
    simp only [FPSoftware.scalar,VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware.scalar,
      FPFused.fusedWeight,VeriTile.Bench.TritonBenchG.QuantizeKvTransform.MixedBits.half_embedding]
    simp only [dot,tensor,accum,FPFused.fusedWeight]

theorem output_isa (mma : MMA16) (rest : Algebra Word) (p : Parameters) (s : State Word) :
    FPFused.outputValue (FPSoftware.scalar rest) (dot mma) p s=output mma p s := by
  funext i
  unfold FPFused.outputValue
  rw [accumulation_isa]
  rfl

theorem per_program (mma : MMA16) (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j) :
    ∃ final, FPFused.exec (FPSoftware.scalar FPCompiled.unused) (dot mma) p s=some final ∧
      FPCompiled.Post p s final (output mma p s) := by
  simpa only [output_isa] using FPFused.correctness
    (FPSoftware.scalar FPCompiled.unused) (dot mma) p s hw

theorem fragment_exact {k : Nat} (h : k/16*16=k) (a : Fin k → Word)
    (j : Nat) (hj : j<k/16) (e : Fin 16) :
    ∃ he : j*16+e.val<k, fragment a j e=a ⟨j*16+e.val,he⟩ := by
  have he : j*16+e.val<k := by
    have hh := Nat.mul_le_mul_right 16 (show j+1≤k/16 by omega)
    rw [Nat.add_mul,h] at hh
    have := e.isLt
    omega
  exact ⟨he,dif_pos he⟩

theorem every_k_lane {k : Nat} (h : k/16*16=k) (i : Fin k) :
    i.val/16<k/16 ∧ i.val%16<16 ∧ i.val/16*16+i.val%16=i.val := by
  refine ⟨?_,Nat.mod_lt _ (by decide),Nat.div_add_mod' _ _⟩
  apply (Nat.div_lt_iff_lt_mul (by decide : 0<16)).mpr
  simpa [h] using i.isLt

#specNonCircular output avoiding [FPFused.exec,FPFused.outputValue,FPFused.accumulation,dot,tensor]
#axiomsClean accumulation_isa
#axiomsClean output_isa
#axiomsClean per_program
#axiomsClean fragment_exact
#axiomsClean every_k_lane
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.ISA
