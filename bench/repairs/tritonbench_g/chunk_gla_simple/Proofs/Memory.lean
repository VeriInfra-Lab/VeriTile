import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.Program

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Memory
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open Program
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def vector (n start : Nat) : Tile .nat [n] := Tile.vec fun i => start+i.val

def matrix (r : RegionName) (b : Nat) (x : Fin m → Nat) (y : Fin n → Nat) : Tile .ptr [m,n] :=
  ⟨fun i => (r,b+x i.1+y i.2.1)⟩

theorem indices_eval (s : BlockState) (name : RegName) (n v : Nat)
    (hv : s.regs .nat [] name = some (Tile.scalar v)) :
    evalOp (indices name n) s = some (vector n (v*n)) := by
  simp only [indices,times,evalOp,evalOp_ref,hv,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem ptr2_eval (s : BlockState) (r : RegionName) (base : Op .nat [])
    (x : Op .nat [m,1]) (y : Op .nat [1,n]) (b : Nat) (xs : Fin m → Nat) (ys : Fin n → Nat)
    (hb : evalOp base s = some (Tile.scalar b))
    (hx : evalOp x s = some ⟨fun i => xs i.1⟩)
    (hy : evalOp y s = some ⟨fun i => ys i.2.1⟩) :
    evalOp (ptr2 r base x y) s = some (matrix r b xs ys) := by
  simp only [ptr2,evalOp,hb,hx,hy,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  change (Region.cast r,0+(b+xs i.1+ys i.2.1)) = (r,b+xs i.1+ys i.2.1)
  simp only [Region.cast_id,Nat.zero_add]

theorem col_eval (s : BlockState) (name : RegName) (n : Nat) (f : Fin n → Nat)
    (hf : s.regs .nat [n] name = some (Tile.vec f)) :
    evalOp (col name n) s = some ⟨fun i => f i.1⟩ := by
  exact evalOp_expandDim_ref_of_regs .nat [n] ⟨1,by simp⟩ name s (Tile.vec f) hf

theorem row_eval (s : BlockState) (name : RegName) (n : Nat) (f : Fin n → Nat)
    (hf : s.regs .nat [n] name = some (Tile.vec f)) :
    evalOp (Program.row name n) s = some ⟨fun i => f i.2.1⟩ := by
  exact evalOp_expandDim_ref_of_regs .nat [n] ⟨0,by simp⟩ name s (Tile.vec f) hf

theorem scaleCol_eval (s : BlockState) (name : RegName) (n stride : Nat) (f : Fin n → Nat)
    (hf : s.regs .nat [n] name = some (Tile.vec f)) :
    evalOp (scaleCol name n stride) s = some ⟨fun i => f i.1*stride⟩ := by
  rw [scaleCol,evalOp_mul,col_eval s name n f hf,evalOp_constNat]
  rfl

theorem scaleRow_eval (s : BlockState) (name : RegName) (n stride : Nat) (f : Fin n → Nat)
    (hf : s.regs .nat [n] name = some (Tile.vec f)) :
    evalOp (scaleRow name n stride) s = some ⟨fun i => f i.2.1*stride⟩ := by
  rw [scaleRow,evalOp_mul,row_eval s name n f hf,evalOp_constNat]
  rfl

theorem mask2_eval (s : BlockState) (x y : RegName) (m n rows cols : Nat)
    (xs : Fin m → Nat) (ys : Fin n → Nat)
    (hx : s.regs .nat [m] x = some (Tile.vec xs))
    (hy : s.regs .nat [n] y = some (Tile.vec ys)) :
    evalOp (mask2 x y m n rows cols) s =
      some ⟨fun i => decide (xs i.1 < rows ∧ ys i.2.1 < cols)⟩ := by
  have hxc := col_eval s x m xs hx
  have hyr := row_eval s y n ys hy
  simp only [mask2,evalOp,hxc,hyr,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  change (decide (xs i.1 < rows) && decide (ys i.2.1 < cols)) = decide (xs i.1 < rows ∧ ys i.2.1 < cols)
  by_cases h1 : xs i.1 < rows <;> by_cases h2 : ys i.2.1 < cols <;> simp [h1,h2]

theorem load2_eval (s : BlockState) (r : RegionName) (ptr x y : RegName)
    (m n rows cols b : Nat) (xs : Fin m → Nat) (ys : Fin n → Nat)
    (ax : Fin m → Nat) (ay : Fin n → Nat)
    (hp : s.regs .ptr [m,n] ptr = some (matrix r b ax ay))
    (hx : s.regs .nat [m] x = some (Tile.vec xs))
    (hy : s.regs .nat [n] y = some (Tile.vec ys)) :
    evalOp (load2 ptr x y m n rows cols) s = some
      ⟨fun i => if xs i.1 < rows ∧ ys i.2.1 < cols
        then some (s.readMem r (b+ax i.1+ay i.2.1)) else some 0⟩ := by
  unfold load2
  rw [evalOp]
  simp only [evalOp_ref,hp,mask2_eval s x y m n rows cols xs ys hx hy,
    Option.bind_eq_bind,Option.bind_some]
  simp only [evalOp,Option.bind_eq_bind,Option.bind_some]
  congr 1
  ext i
  simp only [matrix,Tile.scalar,BlockState.readMemValue_real,decide_eq_true_eq,if_true]

def queryPointer (p : Layout) (l : Launch) (r : Regions) (kb : Nat) :=
  matrix r.query (l.head*p.keyHead) (fun i : Fin p.BT => (l.chunk*p.BT+i.val)*p.keyTime)
    (fun e : Fin p.BK => kb*p.BK+e.val)
def keyPointer (p : Layout) (l : Launch) (r : Regions) (kb : Nat) :=
  matrix r.key (l.head*p.keyHead) (fun e : Fin p.BK => kb*p.BK+e.val)
    (fun j : Fin p.BT => (l.chunk*p.BT+j.val)*p.keyTime)
def historyPointer (p : Layout) (l : Launch) (r : Regions) (kb : Nat) :=
  matrix r.history (l.head*p.historyHead+l.chunk*p.K*p.V)
    (fun e : Fin p.BK => (kb*p.BK+e.val)*p.historyKey)
    (fun j : Fin p.BV => l.value*p.BV+j.val)
def valuePointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrix r (l.head*p.valueHead) (fun i : Fin p.BT => (l.chunk*p.BT+i.val)*p.valueTime)
    (fun j : Fin p.BV => l.value*p.BV+j.val)
def gatePointer (p : Layout) (l : Launch) (r : Regions) : Tile .ptr [p.BT] :=
  ⟨fun i => (r.gate,l.head*p.T+(l.chunk*p.BT+i.1.val))⟩

theorem queryPtr_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (queryPtr p r) s = some (queryPointer p l r kb) :=
  ptr2_eval s _ _ _ _ _ _ _ (mulConst_eval s _ _ _ hh)
    (scaleCol_eval s _ _ _ _ hr) (row_eval s _ _ _ hk)

theorem keyPtr_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (keyPtr p r) s = some (keyPointer p l r kb) :=
  ptr2_eval s _ _ _ _ _ _ _ (mulConst_eval s _ _ _ hh)
    (col_eval s _ _ _ hk) (scaleRow_eval s _ _ _ _ hr)

theorem historyPtr_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (historyPtr p r) s = some (historyPointer p l r kb) := by
  apply ptr2_eval
  · simp only [historyBase,times,evalOp,hh,ht,Option.bind_some,Option.bind_eq_bind]
    rfl
  · exact scaleCol_eval s _ _ _ _ hk
  · exact row_eval s _ _ _ hc

theorem valuePtr_eval (p : Layout) (l : Launch) (r : RegionName) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV))) :
    evalOp (valuePtr p r) s = some (valuePointer p l r) :=
  ptr2_eval s _ _ _ _ _ _ _ (mulConst_eval s _ _ _ hh)
    (scaleCol_eval s _ _ _ _ hr) (row_eval s _ _ _ hc)

theorem gatePtr_eval (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT))) :
    evalOp (gatePtr p r) s = some (gatePointer p l r) := by
  simp only [gatePtr,times,evalOp,hh,hr,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  change (Region.cast r.gate,0+(l.head*p.T+(l.chunk*p.BT+i.1.val))) = _
  simp only [Region.cast_id,Nat.zero_add]
  rfl

theorem queryLoad_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hp : s.regs .ptr [p.BT,p.BK] "p_q" = some (queryPointer p l r kb))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (load2 "p_q" "rows" "keys" p.BT p.BK p.T p.K) s =
      some (queryTile p l (inputs r s) kb) := by
  rw [load2_eval s _ _ _ _ _ _ _ _ _ _ _ _ _ hp hr hk]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp only [queryTile,query,Tile.mat,inputs,keyAddress,Math.row]
  split_ifs <;> rfl

theorem keyLoad_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hp : s.regs .ptr [p.BK,p.BT] "p_k" = some (keyPointer p l r kb))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (load2 "p_k" "keys" "rows" p.BK p.BT p.K p.T) s =
      some (keyTile p l (inputs r s) kb) := by
  rw [load2_eval s _ _ _ _ _ _ _ _ _ _ _ _ _ hp hk hr]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [keyTile,key,Tile.mat,inputs,Math.row]
  split_ifs <;> rfl

theorem historyLoad_eval (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (s : BlockState)
    (hp : s.regs .ptr [p.BK,p.BV] "p_h" = some (historyPointer p l r kb))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV)))
    (hk : s.regs .nat [p.BK] "keys" = some (vector p.BK (kb*p.BK))) :
    evalOp (load2 "p_h" "keys" "cols" p.BK p.BV p.K p.V) s =
      some (historyTile p l (inputs r s) kb) := by
  rw [load2_eval s _ _ _ _ _ _ _ _ _ _ _ _ _ hp hk hc]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [historyTile,history,Tile.mat,inputs,historyAddress,column]
  split_ifs <;> rfl

theorem valueLoad_eval (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (hp : s.regs .ptr [p.BT,p.BV] "p_v" = some (valuePointer p l r.value))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .nat [p.BV] "cols" = some (vector p.BV (l.value*p.BV))) :
    evalOp (load2 "p_v" "rows" "cols" p.BT p.BV p.T p.V) s =
      some (valueTile p l (inputs r s)) := by
  rw [load2_eval s _ _ _ _ _ _ _ _ _ _ _ _ _ hp hr hc]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [valueTile,value,active,Tile.mat,inputs,outputAddress,Math.row,column]
  split_ifs <;> rfl

theorem gateLoad_eval (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (hp : s.regs .ptr [p.BT] "p_g" = some (gatePointer p l r))
    (hr : s.regs .nat [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT))) :
    evalOp (gateLoad p) s = some (gateTile p l (inputs r s)) := by
  simp only [gateLoad,evalOp,hp,hr,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [if_true]
  change (if decide (l.chunk*p.BT+i.val < p.T) then
    s.readMemValue .real r.gate (l.head*p.T+(l.chunk*p.BT+i.val))
    else some 0) = _
  simp only [if_true,decide_eq_true_eq,BlockState.readMemValue_real,
    gateTile,gate,Tile.vec,inputs,gateAddress,Math.row]
  split_ifs <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.Memory
