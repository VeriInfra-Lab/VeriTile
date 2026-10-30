import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelGrid
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelScalar
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Fixed interpretation for operations unreachable in this source. -/
def fallback (zero : α) : Algebra α where
  literal:=fun _ _ _=>zero
  negInf:=zero
  binary:=fun _ _ _ _ _=>zero
  unary:=fun _ _ x=>x
  cast:=fun _ _ _ x=>x
  fromNat:=fun _ _=>zero
  fromInt:=fun _ _=>zero
  fp32Bits:=fun _=>zero
  fp32Load:=id
  reduceMax:=fun _ _ _ _ _ _=>zero
  reduceSum:=fun _ _ _ _ _ _=>zero

abbrev LaneDot (α : Type) := Nat → Nat → (k : Nat) → (Fin k → α) → (Fin k → α) → α → α

def dot (G : LaneDot α) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → α) (b : TileIndex (batch++[k,n]) → α)
    (c : TileIndex (batch++[m,n]) → α) : TileIndex (batch++[m,n]) → α :=
  match batch with
  | []=>fun i=>G i.1.val i.2.1.val k (fun e=>a (i.1,e,PUnit.unit)) (fun e=>b (e,i.2.1,PUnit.unit)) (c i)
  | _::tail=>fun i=>dot G (batch:=tail) (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) (fun j=>c (i.1,j)) i.2
termination_by batch.length

def dotAcc (G : LaneDot α) : DotAcc α := ⟨fun _ _ _ _ _ {batch} {m} {k} {n} a b c=>dot G a b c⟩

theorem foldRange_apply {ι α : Type} (f : Nat → ι → α → α)
    (cur stop stride : Nat) (initial : ι → α) (i : ι) :
    (foldRange (fun q acc j=>f q j (acc j)) cur stop stride initial) i=
      foldRange (fun q x=>f q i x) cur stop stride (initial i) := by
  by_cases hz : stride=0
  · simp [foldRange,hz]
  by_cases hc : cur<stop
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [hz,hc,if_false,if_true]
    exact foldRange_apply f (cur+stride) stop stride _ i
  · simp [foldRange,hz,hc]
termination_by stop-cur
decreasing_by omega

def laneUpdate {α : Type} [Inhabited α] (M : Algebra α) (G : LaneDot α)
    (p : Config) (s : State α) (q : Nat) (i : CIndex p) (acc : α) :=
  G i.1.val i.2.1.val p.BK (fun e=>aValue M p s q (i.1,e,PUnit.unit))
    (fun e=>bValue M p s q (e,i.2.1,PUnit.unit)) acc

theorem output_lane {α : Type} [Inhabited α] (M : Algebra α) (G : LaneDot α)
    (p : Config) (s : State α) (i : CIndex p) :
    outputValue M (dotAcc G) p s i=
      M.cast (some .fp32) .fp16 .real (M.cast none .real .fp16
        (foldRange (fun q=>laneUpdate M G p s q i) 0 p.steps 1 (M.literal (some .fp32) .real 0))) := by
  unfold outputValue accumulation
  have upd : update M (dotAcc G) p s=(fun q acc j=>laneUpdate M G p s q j (acc j)) := by
    funext q acc j
    simp only [update,dotAcc,dot,laneUpdate]
  rw [upd,foldRange_apply]

/-- Coordinate recurrence independent of the source AST, register state,
program identifiers and output storage. The chosen scalar model retains every
padded K lane in ascending source block order. -/
def reference {α : Type} [Inhabited α] (M : Algebra α) (G : LaneDot α)
    (p : Config) (a b : Nat → Nat → α) (r n : Nat) : α :=
  M.cast (some .fp32) .fp16 .real (M.cast none .real .fp16
    (foldRange (fun q acc=>G (r%p.BM) (n%p.BN) p.BK
      (fun e=>M.fp32Load (if q*p.BK+e.val<p.K then a r (q*p.BK+e.val) else M.literal (some .fp32) .real 0))
      (fun e=>M.fp32Load (if q*p.BK+e.val<p.K then b (q*p.BK+e.val) n else M.literal (some .fp32) .real 0)) acc)
      0 (ceil p.K p.BK) 1 (M.literal (some .fp32) .real 0)))

theorem point_reference {α : Type} [Inhabited α] (M : Algebra α) (G : LaneDot α)
    (p : Config) (hc : p.Valid) (s : State α) (w : Work) (hw : p.Active w) :
    point M (dotAcc G) p hc s w=reference M G p
      (fun r k=>read (s.mem "A" (p.A.addr r k)))
      (fun k n=>read (s.mem "B" (p.B.addr k n))) w.1 w.2 := by
  unfold point
  rw [output_lane]
  have upd (q : Nat) (z : α) :
      laneUpdate M G p (atProgram s (owner p w)) q (lane p hc w) z=
        G (w.1%p.BM) (w.2%p.BN) p.BK
          (fun e=>M.fp32Load (if q*p.BK+e.val<p.K then read (s.mem "A" (p.A.addr w.1 (q*p.BK+e.val))) else M.literal (some .fp32) .real 0))
          (fun e=>M.fp32Load (if q*p.BK+e.val<p.K then read (s.mem "B" (p.B.addr (q*p.BK+e.val) w.2)) else M.literal (some .fp32) .real 0)) z := by
    unfold laneUpdate aValue bValue aOffset bOffset aRow bCol
    dsimp only
    rw [own_row,own_col]
    simp only [Nat.mod_eq_of_lt hw.1,Nat.mod_eq_of_lt hw.2,atProgram,BK,lane]
    congr 2 <;> funext e
    all_goals congr 1
    all_goals simp only [Nat.add_comm e.val (q*p.BK)]
    all_goals split_ifs
    all_goals try rfl
    all_goals congr 2
    all_goals simp only [View.addr]
    all_goals ring
  have updates:=funext (fun q=>funext (fun z=>upd q z))
  rw [updates]
  rfl

#axiomsClean foldRange_apply
#axiomsClean output_lane
#axiomsClean point_reference
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelScalar
