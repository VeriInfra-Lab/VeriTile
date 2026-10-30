import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardSource
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul dot rawLoad load scatter)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
abbrev Carry (p : Parameters) (α : Type) := Matrix α p.BK p.BV

def logRate (M : Algebra α) (p : Parameters) (pid : PID) : α :=
  M.unary (some .fp32) .log2 (M.binary (some .fp32) .real .sub (scalar M 1)
    (M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (scalar M (-5))
      (mul M (M.fromNat (some .fp32) (pid.z % p.H)) (scalar M 1)))))
def weight (M : Algebra α) (p : Parameters) (pid : PID) (i : Nat) : α :=
  M.unary (some .fp32) .exp2 (mul M (M.fromInt (some .fp32) ((i:Int)+1)) (logRate M p pid))
def initialValue (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α) (terminal : Bool) : Carry p α :=
  fun i=>if terminal then M.fp32Load
    (if pid.x*p.BK+i.1.val<p.K ∧ pid.y*p.BV+i.2.1.val<p.V then
      (input "terminal" (pid.z*p.K*p.V+(pid.x*p.BK+i.1.val)*p.V+(pid.y*p.BV+i.2.1.val))).read .real
    else zero M) else zero M

def InputEq (a b : Memory α) : Prop := ∀ r : RegionName,r∈(["q","incoming","terminal"] : List RegionName) → ∀ o,a r o=b r o

structure Ready (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α) (s : State α) : Prop where
  inputs : InputEq s.mem input
  ik : s.regs .nat [] "i_k"=some (fun _=>pid.x)
  iv : s.regs .nat [] "i_v"=some (fun _=>pid.y)
  bh : s.regs .nat [] "i_bh"=some (fun _=>pid.z)
  rate : s.regs .real [] "b_b"=some (fun _=>logRate M p pid)
  lane : s.regs .int [p.BT] "o_i"=some (fun i=>(i.1.val:Int))
  weights : s.regs .real [p.BT] "d_i"=some (fun i=>weight M p pid i.1.val)

def HasCarry (p : Parameters) (value : Carry p α) (s : State α) :=
  s.regs .real [p.BK,p.BV] "b_dh"=some value

theorem signed_lane (i : Nat) (hi : i<2^31) :
    (if 2 * (((i:Int)%4294967296).toNat % 4294967296)<4294967296
      then max ((i:Int)%4294967296) 0 % 4294967296
      else max ((i:Int)%4294967296) 0 % 4294967296-4294967296)=(i:Int) := by
  have hn : i<4294967296 := by norm_num at hi; omega
  have hcast : (i:Int)%4294967296=(i:Int) := Int.emod_eq_of_lt (Int.natCast_nonneg _) (by omega)
  have hd : 2*i<4294967296 := by norm_num at hi; omega
  simp [hcast, Nat.mod_eq_of_lt hn, max_eq_left (Int.natCast_nonneg i), hd]

theorem entry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (s : State α) (hBT : p.BT≤2^31) :
    let pid : PID:=⟨s.pids 0,s.pids 1,s.pids 2⟩
    Ready M p pid s.mem (entry M D p scale terminal initial s).val ∧
    HasCarry p (initialValue M p pid s.mem terminal) (entry M D p scale terminal initial s).val := by
  have one : (1.0:ℝ)=1 := by norm_num
  dsimp only
  constructor
  · constructor
    · rw [entry_memory]; intro r hr o; rfl
    · cases terminal <;> simp [entry]
    · cases terminal <;> simp [entry]
    · cases terminal <;> simp [entry]
    · cases terminal <;> simp [entry,logRate,scalar,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,integralMod,Function.comp_def,one]
    · cases terminal <;> simp [entry]
      all_goals
        funext i
        exact signed_lane i.1.val (lt_of_lt_of_le i.1.isLt hBT)
    · cases terminal <;> simp [entry,weight,logRate,scalar,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,integralMod,Function.comp_def,one]
      all_goals
        funext i
        rw [signed_lane i.1.val (lt_of_lt_of_le i.1.isLt hBT)]
  · cases terminal <;> simp [HasCarry,initialValue,entry,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,zero,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]


    all_goals rfl

#axiomsClean entry_ready
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
