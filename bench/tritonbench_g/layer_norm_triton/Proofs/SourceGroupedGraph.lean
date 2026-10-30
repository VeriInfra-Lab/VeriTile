import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGrouped
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardGraph

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedGraph
open VeriTile.Triton FP.Structural SourceModel
open SourceForwardGraph (lit bin cast sum fold_memory_congr)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Tile (α : Type) (B : Nat) := Values α .real [B]
abbrev Carries (α : Type) (B : Nat) := Tile α B × Tile α B

def zero (m : Model α) (B : Nat) : Carries α B := (fun _ => lit m 0,fun _ => lit m 0)

def loaded [Inhabited α] (m : Model α) (s : State α) (r : RegionName) (base N j : Nat) : α :=
  cast m (if j<N then (s.mem r (base+j)).read .real else lit m 0)

def inverse [Inhabited α] (m : Model α) (s : State α) (r : Nat) : α :=
  m.scalar.fp32Load ((s.mem "Rstd" r).read .real)

def xhat [Inhabited α] (m : Model α) (s : State α) (stride N r j : Nat) : α :=
  if j<N then bin m .mul
    (bin m .sub (loaded m s "X" (r*stride) N j) (m.scalar.fp32Load ((s.mem "Mean" r).read .real)))
    (inverse m s r) else lit m 0

def wdy [Inhabited α] (m : Model α) (s : State α) (stride N r j : Nat) : α :=
  if j<N then bin m .mul
    (cast m (if j<N then (s.mem "W" j).read .real else s.undef .real "W" j))
    (loaded m s "DY" (r*stride) N j) else lit m 0

def first [Inhabited α] (m : Model α) (s : State α) (stride N B r : Nat) : α :=
  bin m .div (sum m B (fun j => bin m .mul (xhat m s stride N r j.val) (wdy m s stride N r j.val))) (lit m N)

def second [Inhabited α] (m : Model α) (s : State α) (stride N B r : Nat) : α :=
  bin m .div (sum m B (fun j => wdy m s stride N r j.val)) (lit m N)

def dx [Inhabited α] (m : Model α) (s : State α) (stride N B r j : Nat) : α :=
  cast m (bin m .mul (bin m .sub (wdy m s stride N r j)
    (bin m .add (bin m .mul (xhat m s stride N r j) (first m s stride N B r))
      (second m s stride N B r))) (inverse m s r))

def nextCarries [Inhabited α] (m : Model α) (s : State α) (stride N B group r : Nat)
    (a : Carries α B) : Carries α B :=
  (fun i => cast m (let p := cast m (bin m .mul (loaded m s "DY" (r*stride) N i.1.val)
      (xhat m s stride N r i.1.val)); if r=group then p else bin m .add p (a.1 i)),
   fun i => cast m (let p := cast m (loaded m s "DY" (r*stride) N i.1.val);
      if r=group then p else bin m .add p (a.2 i)))

def storeDX [Inhabited α] (m : Model α) (s : State α) (stride N B r : Nat) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i => if i.1.val<N then
    t.write "DX" (r*stride+i.1.val) (.mk .real (dx m s stride N B r i.1.val)) else t) s

def Invariant (s : State α) (N B group : Nat) (a : Carries α B) (t : State α) : Prop :=
  t.mem=s.mem ∧ t.undef=s.undef ∧
  t.regs .nat [] "group"=some (fun _ => group) ∧
  t.regs .nat [B] "cols"=some (fun i => i.1.val) ∧
  t.regs .bool [B] "mask"=some (fun i => decide (i.1.val<N)) ∧
  t.regs .ptr [B] "DW"=some (fun i : TileIndex [B] => (("DW" : RegionName),group*N+i.1.val)) ∧
  t.regs .ptr [B] "DB"=some (fun i : TileIndex [B] => (("DB" : RegionName),group*N+i.1.val)) ∧
  t.regs .real [B] "dw_acc"=some a.1 ∧ t.regs .real [B] "db_acc"=some a.2

@[simp] theorem masked_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).undef=s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons]; split <;> exact ih _

theorem prelude_refines [Inhabited α] (m : Model α) (s : State α) (stride M N G B : Nat) :
    ∃ t,run (engine m) (SourceGrouped.prelude stride M N G B) s=some t ∧
      Invariant s N B (s.pids 0) (zero m B) t := by
  simp [SourceGrouped.prelude,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Invariant,zero,lit,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]
  refine ⟨?_,?_,?_⟩ <;> funext i <;> simp [bop]

theorem body_refines [Inhabited α] (m : Model α) (s t : State α) (stride M N G B group r : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) :
    ∃ u,run (engine m) (SourceGrouped.body stride M N G B) (t.setReg "row" .nat [] (fun _ => r))=some u ∧
      Invariant (storeDX m s stride N B r) N B group (nextCarries m s stride N B group r a) u := by
  obtain ⟨hm,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  by_cases hr : r=group <;>
    simp [SourceGrouped.body,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Invariant,
      engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
      State.setReg,FP.Structural.store,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb,hm,hu,hr,
      show (0.0 : ℝ)=0 by norm_num]
  all_goals simp only [storeDX,nextCarries,dx,first,second,xhat,wdy,inverse,loaded,lit,bin,
    SourceForwardGraph.cast,sum,hr]
  all_goals simp only [ite_true,ite_false,masked_undef,true_and,and_true]
  all_goals
    apply fold_memory_congr
    · rfl
    · intro u v i huv
      by_cases hi : i.1.val<N <;> simp [hi,State.write,huv] <;> rfl

#axiomsClean prelude_refines
#axiomsClean body_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedGraph
