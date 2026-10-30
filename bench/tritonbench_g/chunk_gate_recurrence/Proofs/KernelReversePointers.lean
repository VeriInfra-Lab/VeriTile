import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReversePointers
open VeriTile.Triton FP.Structural KernelProgram
noncomputable section
variable {α : Type} [Inhabited α]

def off (c : Shape) (p : Program c) (start n : Nat) (e : Fin 64) (j : Fin 16) : Nat :=
  address c p start e.val j.val-n*(c.K*c.V)
def gate (c : Shape) (p : Program c) (n : Nat) := gateAddress c p (c.N-1)-n
def gradient (c : Shape) (p : Program c) (n : Nat) := gradientAddress c p (c.N-2)-n*(c.KT*c.VT)

theorem subtract_stride (base start n stride : Nat) (hn : n≤start) :
    base+start*stride-n*stride=base+(start-n)*stride := by
  rw [Nat.sub_mul,Nat.add_sub_assoc (Nat.mul_le_mul_right stride hn)]
theorem live (c : Shape) (p : Program c) (start n : Nat) (hn : n≤start)
    (e : Fin 64) (j : Fin 16) : off c p start n e j=address c p (start-n) e.val j.val := by
  unfold off address layout launch TritonBenchG.ChunkGateRecurrence.RealProgram.address
  simp only [Nat.mul_assoc]
  exact subtract_stride _ _ _ _ hn
theorem gate_live (c : Shape) (p : Program c) (n : Nat) (hn : n≤c.N-1) :
    gate c p n=gateAddress c p (c.N-1-n) := by
  simp only [gate,gateAddress,Nat.add_sub_assoc hn]
theorem gradient_live (c : Shape) (p : Program c) (n : Nat) (hn : n≤c.N-2) :
    gradient c p n=gradientAddress c p (c.N-2-n) := by
  simp only [gradient,gradientAddress,Nat.mul_assoc]
  exact subtract_stride _ _ _ _ hn
theorem off_next (c : Shape) (p : Program c) (start n : Nat) :
    (fun e j => off c p start n e j-c.K*c.V)=off c p start (n+1) := by
  funext e j
  simp [off,Nat.sub_sub,Nat.add_mul]
theorem gate_next (c : Shape) (p : Program c) (n : Nat) :
    gate c p n-1=gate c p (n+1) := by simp [gate,Nat.sub_sub]
theorem gradient_next (c : Shape) (p : Program c) (n : Nat) :
    gradient c p n-c.KT*c.VT=gradient c p (n+1) := by simp [gradient,Nat.sub_sub,Nat.add_mul]

structure Fixed (c : Shape) (p : Program c) (u : State α) : Prop where
  head : u.regs .nat [] "offset_bh"=some (fun _ => p.head)
  key : u.regs .nat [] "offset_d"=some (fun _ => p.key)
  value : u.regs .nat [] "offset_s"=some (fun _ => p.value)
  numK : u.regs .nat [] "NUM_K"=some (fun _ => c.KT)
  numV : u.regs .nat [] "NUM_V"=some (fun _ => c.VT)
  offsK : u.regs .nat [64] "offs_k"=some (fun i => i.1.val)
  offsV : u.regs .nat [16] "offs_v"=some (fun i => i.1.val)

#axiomsClean live
#axiomsClean gradient_live
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReversePointers
