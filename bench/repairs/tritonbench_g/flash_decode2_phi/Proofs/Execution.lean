import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.SourceConnection
import VeriTile.Triton
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

abbrev lane {BD : Nat} (j : TileIndex [BD]) : Nat := j.1.val
@[simp] theorem tile_eta {BD : Nat} (j : TileIndex [BD]) : (j.1,PUnit.unit)=j := by
  rcases j with ⟨j,⟨⟩⟩; rfl

def count (s : State α) (Q : Nat) : Nat :=
  let l := (s.mem "L" (s.pids 0)).read .nat
  (if l≤0 then 0 else l+Q-1)/Q

def voff (s : State α) (H S D j : Nat) (k : TileIndex [BD]) : Nat :=
  s.pids 0*(H*S*D)+s.pids 1*(S*D)+lane k+j*D

def eoff (s : State α) (H S j : Nat) : Nat := s.pids 0*(H*S)+s.pids 1*S+j

def ooff (s : State α) (H D : Nat) (k : TileIndex [BD]) : Nat :=
  s.pids 0*(H*D)+s.pids 1*D+lane k

def active (masked : Bool) (D : Nat) (k : TileIndex [BD]) : Prop :=
  masked=Bool.false ∨ lane k<D
instance (masked : Bool) (D : Nat) (k : TileIndex [BD]) : Decidable (active masked D k) :=
  inferInstanceAs (Decidable (masked=Bool.false ∨ lane k<D))

def zero (A : Algebra α) : α := A.literal (some .fp32) .real 0
def add (A : Algebra α) : α → α → α := A.binary (some .fp32) .real .add
def sub (A : Algebra α) : α → α → α := A.binary (some .fp32) .real .sub
def mul (A : Algebra α) : α → α → α := A.binary (some .fp32) .real .mul
def divide (A : Algebra α) : α → α → α := A.binary (some .fp32) .real .div
def maximum (A : Algebra α) : α → α → α := A.binary (some .fp32) .real .max
def exponential (A : Algebra α) : α → α := A.unary (some .fp32) .exp

def value (A : Algebra α) (s : State α) (masked : Bool) (H S D j : Nat)
    (k : TileIndex [BD]) : α :=
  A.fp32Load (if active masked D k then (s.mem "M" (voff s H S D j k)).read .real else zero A)
def logvalue (A : Algebra α) (s : State α) (H S j : Nat) : α :=
  A.fp32Load ((s.mem "E" (eoff s H S j)).read .real)

structure Carry (α : Type) (BD : Nat) where
  m : α
  z : α
  a : TileIndex [BD] → α

def initial (A : Algebra α) (BD : Nat) : Carry α BD := ⟨A.negInf,A.literal none .real 0,fun _ => zero A⟩
def next (A : Algebra α) (s : State α) (masked : Bool) (H S D j : Nat)
    (c : Carry α BD) : Carry α BD :=
  let x := logvalue A s H S j
  let m := maximum A x c.m
  let old := exponential A (sub A c.m m)
  let fresh := exponential A (sub A x m)
  ⟨m,add A (mul A c.z old) fresh,
    fun k => add A (mul A (c.a k) old) (mul A fresh (value A s masked H S D j k))⟩
def values (A : Algebra α) (s : State α) (masked : Bool) (H S D BD : Nat) : Nat → Carry α BD
  | 0 => initial A BD
  | j+1 => next A s masked H S D j (values A s masked H S D BD j)

def scatter (s : State α) (H D BD : Nat) (masked : Bool) (v : TileIndex [BD] → α) : State α :=
  (TileShape.allIndices [BD]).foldl (fun q k => if active masked D k then
    q.write "O" (ooff s H D k) (.mk .real (v k)) else q) s

structure Inv (A : Algebra α) (s : State α) (masked : Bool) (H S D Q BD j : Nat)
    (q : State α) : Prop where
  mem : q.mem=s.mem
  pids : q.pids=s.pids
  batch : q.regs .nat [] "cur_batch"=some (fun _ => s.pids 0)
  head : q.regs .nat [] "cur_head"=some (fun _ => s.pids 1)
  lanes : q.regs .nat [BD] "offs_d"=some lane
  stop : q.regs .nat [] "block_n_size"=some (fun _ => count s Q)
  ov : q.regs .nat [BD] "offs_v"=some (fun k => voff s H S D 0 k)
  oe : q.regs .nat [] "offs_logic"=some (fun _ => eoff s H S 0)
  max : q.regs .real [] "max_logic"=some (fun _ => (values A s masked H S D BD j).m)
  sum : q.regs .real [] "sum_exp"=some (fun _ => (values A s masked H S D BD j).z)
  acc : q.regs .real [BD] "acc"=some (values A s masked H S D BD j).a

theorem prologue_run (A : Algebra α) (s : State α) (masked : Bool) (H S D Q BD : Nat) :
    ∃ q, run A (prologue H S D Q BD) s=some q ∧ Inv A s masked H S D Q BD 0 q := by
  simp [prologue,phi,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
    natLt,natLe,numericLt,numericLe,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  constructor
  all_goals simp [Inv,values,initial,zero,voff,eoff,lane,count,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv]
  all_goals funext k
  all_goals first | rfl | (try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv])
  all_goals norm_num

theorem body_run (A : Algebra α) (s q : State α) (masked : Bool) (H S D Q BD j : Nat)
    (I : Inv A s masked H S D Q BD j q) :
    ∃ r, run A (body masked H S D Q BD)
      (q.setReg "block_seq_n" .nat [] (fun _ => j))=some r ∧
      Inv A s masked H S D Q BD (j+1) r := by
  rcases I with ⟨hm,hp,hb,hh,hl,hs,hv,he,hmax,hsum,ha⟩
  cases masked <;>
    simp [body,kernel,phi,llama,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,natLe,numericLt,numericLe,Region.cast,State.setReg,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
      hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp]
  all_goals constructor
  all_goals simp [Inv,values,next,maximum,exponential,sub,mul,add,logvalue,value,active,zero,
    voff,eoff,lane,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp]
  all_goals funext k
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals norm_num


theorem range_complete (A : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (stop : Nat) (P : Nat → State α → Prop)
    (one : ∀ i, i<stop → ∀ s, P i s → ∃ q,
      run A body (s.setReg idx .nat [] (fun _ => i))=some q ∧ P (i+1) q)
    (s : State α) (start : Nat) (bound : start≤stop) (initial : P start s) :
    ∃ q, range A idx start stop 1 body s=some q ∧ P stop q := by
  have go : ∀ n i, i+n=stop → ∀ s, P i s →
      ∃ q, range A idx i stop 1 body s=some q ∧ P stop q := by
    intro n
    induction n with
    | zero =>
      intro i hi s hs
      have h : i=stop := by omega
      subst i
      exact ⟨s,by rw [range];simp,hs⟩
    | succ n ih =>
      intro i hi s hs
      have hlt : i<stop := by omega
      obtain ⟨q,hq,hP⟩ := one i hlt s hs
      obtain ⟨r,hr,hR⟩ := ih (i+1) (by omega) q hP
      refine ⟨r,?_,hR⟩
      rw [range]
      simpa [hlt,hq] using hr
  exact go (stop-start) start (by omega) s initial

theorem loop_run (A : Algebra α) (s : State α) (masked : Bool) (H S D Q BD : Nat) :
    ∃ q, run A (prologue H S D Q BD ++
      [.forRangeDyn "block_seq_n" (.constNat 0) (.ref .nat [] "block_n_size")
        (.constNat 1) (body masked H S D Q BD)]) s=some q ∧
      Inv A s masked H S D Q BD (count s Q) q := by
  obtain ⟨q,hq,I⟩ := prologue_run A s masked H S D Q BD
  obtain ⟨r,hr,J⟩ := range_complete A "block_seq_n" (body masked H S D Q BD)
    (count s Q) (Inv A s masked H S D Q BD)
    (fun j _ q h => body_run A s q masked H S D Q BD j h) q 0 (by omega) I
  refine ⟨r,?_,J⟩
  rw [run_append,hq]
  simp [run,step,evalOp_unfold,Option.bind,I.stop,hr]

def output (A : Algebra α) (s : State α) (masked : Bool) (H S D Q BD : Nat)
    (k : TileIndex [BD]) : α :=
  let c := values A s masked H S D BD (count s Q)
  divide A (c.a k) c.z

theorem scatter_memory_congr {ι : Type} (l : List ι) (s u : State α)
    (R : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → α) (h : s.mem=u.mem) :
    (l.foldl (fun v i => if active i then v.write R (offset i) (.mk .real (value i)) else v) s).mem =
    (l.foldl (fun v i => if active i then v.write R (offset i) (.mk .real (value i)) else v) u).mem := by
  induction l generalizing s u with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      apply ih
      simp only [State.write,h]
    · simp only [hi, ↓reduceIte]
      exact ih s u h

theorem unmasked_memory_congr {ι : Type} (l : List ι) (s u : State α)
    (R : RegionName) (offset : ι → Nat) (value : ι → α) (h : s.mem=u.mem) :
    (l.foldl (fun v i => v.write R (offset i) (.mk .real (value i))) s).mem =
    (l.foldl (fun v i => v.write R (offset i) (.mk .real (value i))) u).mem := by
  simpa only [ite_true] using scatter_memory_congr l s u R offset (fun _ => True) value h

theorem epilogue_run (A : Algebra α) (s q : State α) (masked : Bool) (H S D Q BD : Nat)
    (I : Inv A s masked H S D Q BD (count s Q) q) :
    ∃ r, run A (epilogue masked H S D Q BD) q=some r ∧
      r.mem=(scatter s H D BD masked (output A s masked H S D Q BD)).mem := by
  rcases I with ⟨hm,hp,hb,hh,hl,hs,hv,he,hmax,hsum,ha⟩
  cases masked <;>
    simp [epilogue,kernel,phi,llama,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,natLe,numericLt,numericLe,Region.cast,State.setReg,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
      hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp,store]
  all_goals simp [scatter,active,output,divide,ooff,lane]
  · exact unmasked_memory_congr _ _ _ _ _ _ hm
  · exact scatter_memory_congr _ _ _ _ _ _ _ hm


theorem complete (A : Algebra α) (s : State α) (masked : Bool) (H S D Q BD : Nat) :
    ∃ q, FP.Structural.exec A (kernel masked H S D Q BD) s=some q ∧
      q.mem=(scatter s H D BD masked (output A s masked H S D Q BD)).mem := by
  obtain ⟨q,hq,I⟩ := loop_run A s masked H S D Q BD
  obtain ⟨r,hr,hm⟩ := epilogue_run A s q masked H S D Q BD I
  refine ⟨r,?_,hm⟩
  change run A (kernel masked H S D Q BD).surfaceBody s=some r
  rw [decomposition,run_append,hq]
  simpa using hr

theorem output_offset_injective (s : State α) (H D BD : Nat) :
    Function.Injective (ooff s H D (BD:=BD)) := by
  intro i j h
  have hf : i.1=j.1 := by apply Fin.ext; dsimp [ooff,lane] at h; omega
  rcases i with ⟨i,⟨⟩⟩
  rcases j with ⟨j,⟨⟩⟩
  simp_all

theorem scatter_readback (s : State α) (H D BD : Nat) (masked : Bool)
    (v : TileIndex [BD] → α) (k : TileIndex [BD]) (hk : active masked D k) :
    (scatter s H D BD masked v).mem "O" (ooff s H D k)=.mk .real (v k) := by
  exact State.masked_scatter_readback "O" (ooff s H D)
    (fun j => .mk .real (v j)) (active masked D) s k hk (output_offset_injective s H D BD)

theorem scatter_frame (s : State α) (H D BD : Nat) (masked : Bool)
    (v : TileIndex [BD] → α) (r : RegionName) (a : Nat)
    (miss : r≠("O" : RegionName) ∨ ∀ k : TileIndex [BD], active masked D k → a≠ooff s H D k) :
    (scatter s H D BD masked v).mem r a=s.mem r a := by
  apply State.masked_scatter_frame
  rcases miss with h|h
  · exact Or.inl h
  · exact Or.inr (fun k _ hk => h k hk)

#axiomsClean complete
#axiomsClean scatter_readback
#axiomsClean scatter_frame

end
end VeriTile.Bench.Repairs.FlashDecode2
