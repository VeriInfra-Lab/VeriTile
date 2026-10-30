import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPReference

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPCarry
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault FPReference
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def dPointers (initial : State α) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (step : Nat) : Values α .ptr [BM,BCS] := fun i =>
  (p.Dout,dBase initial p + (row initial BM BN p i.1*p.dColumn+
    (step*BCS+i.2.1.val)*p.dRow))

def aPointers (initial : State α) (BN BCS : Nat) (p : RealProgram.Parameters)
    (step : Nat) : Values α .ptr [BCS,BN] := fun i =>
  (p.A,aBase initial p+((step*BCS+i.1.val)*p.aRow+column initial BN p i.2.1*p.aColumn))

structure Holds (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (stop step : Nat) (s : State α) : Prop where
  live : Live (backend M D) BM BN BCS p initial s
  count : step ≤ stop
  dout : s.regs .ptr [BM,BCS] "dout_ptrs" = some (dPointers initial BM BN BCS p step)
  a : s.regs .ptr [BCS,BN] "a_ptrs" = some (aPointers initial BN BCS p step)
  acc : s.regs .real [BM,BN] "acc" = some (carry M D BM BN BCS p initial step)

theorem initial (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (s : State α) (stop : Nat) :
    Holds M D BM BN BCS p s stop 0 (setup (backend M D) BM BN BCS p s).val := by
  refine ⟨live_initial _ _ _ _ _ _,Nat.zero_le _,?_,?_,?_⟩
  · simp [setup]
    funext i
    simp [dPointers,dBase,row,chunk,head,backend,bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]
  · simp [setup]
    funext i
    simp [aPointers,aBase,column,chunk,head,backend,bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_assoc]
  · simp [setup,carry,backend]

theorem iteration (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial s : State α) (stop index : Nat)
    (hi : index < stop) (h : Holds M D BM BN BCS p initial stop index s) :
    ∃ t, run (engine (backend M D)) (parts BM BN BCS p).body
      (s.setReg "cs" .nat [] (fun _ => index)) = some t ∧
      Holds M D BM BN BCS p initial stop (index+1) t := by
  rcases s with ⟨sm,sr,sp,snp,su⟩
  have he : sm = initial.mem := h.live.mem
  subst sm
  let s : State α := ⟨initial.mem,sr,sp,snp,su⟩
  let entry := s.setReg "cs" .nat [] (fun _ => index)
  have hr : entry.regs .nat [BM] "offs_m" = some (fun i => row initial BM BN p i.1) := by
    simpa [entry,setup,row] using h.live.fixed .nat [BM] "offs_m" (by decide)
  have hn : entry.regs .nat [BN] "offs_n" = some (fun i => column initial BN p i.1) := by
    simpa [entry,setup,column] using h.live.fixed .nat [BN] "offs_n" (by decide)
  have hk : entry.regs .nat [BCS] "offs_cs" = some (fun i => i.1.val) := by
    simpa [entry,setup] using h.live.fixed .nat [BCS] "offs_cs" (by decide)
  have hl : entry.regs .nat [] "chunk_size_limit" = some (fun _ => length initial p) := by
    have he := h.live.fixed .nat [] "chunk_size_limit" (by decide)
    by_cases hc : p.chunk < p.seqlen-initial.pids 2/p.groups*p.chunk
    · simpa [entry,setup,length,chunk,hc,Nat.min_eq_left (Nat.le_of_lt hc)] using he
    · simpa [entry,setup,length,chunk,hc,Nat.min_eq_right (Nat.le_of_not_gt hc)] using he
  let result := FPDefault.iteration M D BM BN BCS p entry
    (dPointers initial BM BN BCS p index) (aPointers initial BN BCS p index)
    (fun i => row initial BM BN p i.1) (fun i => column initial BN p i.1)
    (fun i => i.1.val) (length initial p) index (carry M D BM BN BCS p initial index)
    (by simpa [entry] using h.dout) (by simpa [entry] using h.a) hr hn hk hl
    (by simp [entry]) (by simpa [entry] using h.acc)
  obtain ⟨t,ht,hLive⟩ := live_iteration M D BM BN BCS p initial s index h.live
  have he : t = result.val := by
    exact Option.some.inj (ht.symm.trans result.property)
  subst t
  refine ⟨result.val,result.property,hLive,Nat.succ_le_of_lt hi,?_,?_,?_⟩
  · simp [result,FPDefault.iteration]
    congr 1
    funext i
    simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,dPointers,Prod.mk.injEq,true_and]
    ring
  · simp [result,FPDefault.iteration]
    congr 1
    funext i
    simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,aPointers,Prod.mk.injEq,true_and]
    ring
  · simp [result,FPDefault.iteration]
    congr 1
    funext i
    simp only [carry,backend,entry,State.setReg_mem,h.live.mem,bop,Broadcast.leftIndex,Broadcast.rightIndex]
    congr 1
    congr 1
    · funext k
      norm_num [leftBlock,dPointers,Cell.read,s]
      rfl
    · funext k
      norm_num [rightBlock,aPointers,Cell.read,s]
      rfl

theorem loop (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (s : State α) (stop : Nat) :
    ∃ t, range (engine (backend M D)) "cs" 0 stop 1 (parts BM BN BCS p).body
      (setup (backend M D) BM BN BCS p s).val = some t ∧
      Holds M D BM BN BCS p s stop stop t := by
  obtain ⟨last,t,ht,hstop,hInv⟩ := FP.Structural.range_invariant
    (engine (backend M D)) (idx := "cs") (stop := stop) (stride := 1)
    (body := (parts BM BN BCS p).body) (P := Holds M D BM BN BCS p s stop)
    (by decide) (fun i s hi h => iteration M D BM BN BCS p _ s stop i hi h)
    0 (setup (backend M D) BM BN BCS p s).val (initial M D BM BN BCS p s stop)
  have he : last = stop := Nat.le_antisymm hInv.count hstop
  subst last
  exact ⟨t,ht,hInv⟩

#axiomsClean initial
#axiomsClean iteration
#axiomsClean loop
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPCarry
