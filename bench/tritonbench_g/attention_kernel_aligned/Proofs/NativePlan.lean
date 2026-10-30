import bench.tritonbench_g.attention_kernel_aligned.Proofs.Domain
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan
open VeriTile Triton Full Whole
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 16000000

/-- Integer routing tree adapted from attention_kernel Whole002.NativePlan.
No arithmetic or final-result equation is present. -/
inductive Tree where
  | leaf (index : Nat)
  | merge (a b : Tree)
  | shuffle (mask : Nat) (a : Tree)
  | shared (slot : Nat) (a : Tree)
def Tree.leaves : Tree → List Nat
  | .leaf i=>[i]
  | .merge a b=>a.leaves++b.leaves
  | .shuffle _ a=>a.leaves
  | .shared _ a=>a.leaves

inductive Width where
  | n16 | n32 | n64 | n128
  deriving DecidableEq, Repr

def Width.value : Width → Nat
  | .n16 => 16 | .n32 => 32 | .n64 => 64 | .n128 => 128

inductive WarpColumns where
  | w1 | w2 | w4 | w8
  deriving DecidableEq, Repr

def WarpColumns.value : WarpColumns → Nat
  | .w1 => 1 | .w2 => 2 | .w4 => 4 | .w8 => 8

def warps (width : Width) (columns : WarpColumns) : Nat :=
  min columns.value (width.value/8)

def localIndex (width : Width) (columns : WarpColumns) (warp lane j : Nat) : Nat :=
  j/2*(8*warps width columns)+warp*8+2*lane+j%2

def localTree (width : Width) (columns : WarpColumns) (warp lane : Nat) : Tree :=
  (List.range (width.value/(4*warps width columns)-1)).foldl
    (fun t j=>.merge t (.leaf (localIndex width columns warp lane (j+1))))
    (.leaf (localIndex width columns warp lane 0))

def butterfly (masks : List Nat) (input : Nat→Tree) : Nat→Tree :=
  masks.foldl (fun prev mask lane=>.merge (prev lane) (.shuffle mask (prev (Nat.xor lane mask)))) input

def warpTree (width : Width) (columns : WarpColumns) (warp lane : Nat) : Tree :=
  butterfly [2,1] (localTree width columns warp) lane

/-- With one unique column warp, every lane keeps its own ordered butterfly
result. With multiple column warps, lane zero stores each partial; the shared
cross-warp reduction and final shared broadcast fix one common tree. -/
def tree (width : Width) (columns : WarpColumns) (owner : Fin 32) : Tree :=
  if warps width columns=1 then warpTree width columns 0 (owner.val%4)
  else .shared (warps width columns)
    (butterfly ([4,2,1].filter (·<warps width columns))
      (fun warp=>.shared warp (warpTree width columns warp 0)) 0)

def physicalOwner (width : Width) (columns : WarpColumns) (column : Nat) : Fin 32 :=
  ⟨((column/8)%warps width columns)*4+(column%8)/2,by
    have hw : 0<warps width columns ∧ warps width columns≤8 := by cases width <;> cases columns <;> decide
    have h1:=Nat.mod_lt (column/8) hw.1
    have h2:=Nat.mod_lt column (by decide : 0<8)
    omega⟩

theorem reduction_coverage (width : Width) (columns : WarpColumns) (u : Fin 32) :
    (tree width columns u).leaves.Perm (List.range width.value) := by
  have h : ∀ v : Fin 4,
      (tree width columns ⟨v.val,by omega⟩).leaves.Perm (List.range width.value) := by
    cases width <;> cases columns <;> decide
  simpa only [tree,Nat.mod_mod] using h ⟨u.val%4,Nat.mod_lt _ (by decide)⟩

/-- Exact logical coordinates of the Ampere m16n8k16 packed fragments.
The implementation of shared/ldmatrix routing to these finite coordinates is
an ordinary compiler boundary; no value equation is hidden in this table. -/
def aCoordinate (lane : Fin 32) (register : Fin 4) (half : Fin 2) : Nat×Nat :=
  (lane.val/4+8*(register.val%2),2*(lane.val%4)+8*(register.val/2)+half.val)

def bCoordinate (lane : Fin 32) (register : Fin 2) (half : Fin 2) : Nat×Nat :=
  (2*(lane.val%4)+8*register.val+half.val,lane.val/4)

def cCoordinate (lane : Fin 32) (register : Fin 4) : Nat×Nat :=
  (lane.val/4+8*(register.val/2),2*(lane.val%4)+register.val%2)

theorem packed_bounds (lane : Fin 32) (a : Fin 4) (b h : Fin 2) :
    (aCoordinate lane a h).1<16 ∧ (aCoordinate lane a h).2<16 ∧
    (bCoordinate lane b h).1<16 ∧ (bCoordinate lane b h).2<8 ∧
    (cCoordinate lane a).1<16 ∧ (cCoordinate lane a).2<8 := by
  have hl:=lane.isLt;have ha:=a.isLt;have hb:=b.isLt;have hh:=h.isLt
  simp only [aCoordinate,bCoordinate,cCoordinate]
  omega

/-- Every fixed instruction output coordinate has exactly one lane/register. -/
theorem output_coordinate (row : Fin 16) (col : Fin 8) :
    let lane : Fin 32 := ⟨(row.val%8)*4+col.val/2,by omega⟩
    let reg : Fin 4 := ⟨(row.val/8)*2+col.val%2,by omega⟩
    cCoordinate lane reg=(row.val,col.val) := by
  have hr:=row.isLt;have hc:=col.isLt
  simp only [cCoordinate]
  congr 1 <;> omega


def widthOf : Nat → Width
  | 16=>.n16 | 32=>.n32 | 128=>.n128 | _=>.n64
def columnsOf : Nat → WarpColumns
  | 2=>.w2 | 4=>.w4 | 8=>.w8 | _=>.w1

theorem widthOf_value (n : Nat) (h : n∈[16,32,64,128]) : (widthOf n).value=n := by
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h <;> subst n <;> rfl

def selected (p : Parameters) (owner : Fin 4) : Tree :=
  tree (widthOf p.blockN) (columnsOf p.nativeColumns) ⟨owner.val,by omega⟩

theorem selected_coverage (p : Parameters) (s : Selected p) (owner : Fin 4) :
    (selected p owner).leaves.Perm (List.range p.blockN) := by
  unfold selected
  have h:=reduction_coverage (widthOf p.blockN) (columnsOf p.nativeColumns) ⟨owner.val,by omega⟩
  simpa only [widthOf_value p.blockN s.blockN] using h

/-- The duplicated BF16 normalizer graph at BM128/W4 uses one thread per
row, with an increasing serial fold and no shuffle. -/
def serialTree (w : Width) : Tree :=
  (List.range (w.value-1)).foldl (fun t j=>.merge t (.leaf (j+1))) (.leaf 0)
theorem serial_coverage (w : Width) : (serialTree w).leaves.Perm (List.range w.value) := by
  cases w <;> decide

structure Routes where
  a : Fin 32→Fin 4→Fin 2→Nat×Nat
  b : Fin 32→Fin 2→Fin 2→Nat×Nat
  c : Fin 32→Fin 4→Nat×Nat
  bias : Fin 32→Fin 4→Nat×Nat
  localSource : Fin 8→Fin 4→Fin 32→Nat
  shuffle : Fin 32→Fin 3→Nat
  shared : Fin 8→Nat
  broadcast : Fin 32→Nat
  qkChunk : Fin 8→Fin 16→Nat
  pvChunk : Fin 8→Fin 16→Nat

def mask (i : Fin 3) : Nat := [4,2,1][i.val]

def LocalRouting (p : Parameters) (r : Routes) : Prop :=
  (∀ lane reg half,r.a lane reg half=aCoordinate lane reg half) ∧
  (∀ lane reg half,r.b lane reg half=bCoordinate lane reg half) ∧
  (∀ lane reg,r.c lane reg=cCoordinate lane reg) ∧
  (∀ lane reg,r.bias lane reg=cCoordinate lane reg) ∧
  (∀ warp lane j,r.localSource warp lane j=localIndex (widthOf p.blockN) (columnsOf p.nativeColumns) warp.val lane.val j.val) ∧
  (∀ lane i,r.shuffle lane i=Nat.xor lane.val (mask i)) ∧
  (∀ warp,r.shared warp=warp.val) ∧
  (∀ owner,r.broadcast owner=0) ∧
  (∀ block k,r.qkChunk block k=block.val*16+k.val) ∧
  (∀ block k,r.pvChunk block k=block.val*16+k.val)

def canonicalRoutes (p : Parameters) : Routes :=
  ⟨aCoordinate,bCoordinate,cCoordinate,cCoordinate,
    (fun warp lane j=>localIndex (widthOf p.blockN) (columnsOf p.nativeColumns) warp.val lane.val j.val),
    (fun lane i=>Nat.xor lane.val (mask i)),(fun warp=>warp.val),(fun _=>0),
    (fun block k=>block.val*16+k.val),(fun block k=>block.val*16+k.val)⟩
theorem canonical_routing (p : Parameters) : LocalRouting p (canonicalRoutes p) := by
  simp [LocalRouting,canonicalRoutes]

theorem chunk_coverage (n : Nat) (hn : n∈[16,32,64,128]) (k : Fin n) :
    k.val/16<n/16 ∧ k.val/16*16+k.val%16=k.val := by
  have hk:=k.isLt
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hn
  rcases hn with h|h|h|h <;> subst n <;> omega

#axiomsClean reduction_coverage
#axiomsClean selected_coverage
#axiomsClean serial_coverage
#axiomsClean packed_bounds
#axiomsClean output_coordinate
#axiomsClean canonical_routing
#axiomsClean chunk_coverage
#specNonCircular LocalRouting avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan
