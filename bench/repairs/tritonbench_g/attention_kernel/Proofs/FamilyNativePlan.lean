import bench.repairs.tritonbench_g.attention_kernel.Proofs.Native
import bench.repairs.tritonbench_g.attention_kernel.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionKernel.Whole001
namespace NativePlan
open Native
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 16000000

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

/-- Finite source-specific numerical layout selection. This is data, not a
dot, reduction, execution or output equation. Pipeline stages, pointer/stride
alignment and JIT specialization may select a different member of this same
family. Successful compiler local routing binds the selected layout. -/
structure Selection (p : Source.Parameters) (abi : Whole.InputABI p) where
  width : Width
  width_eq : width.value=p.bn
  columns : WarpColumns
  columns_bound : columns.value≤abi.numWarps

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
  cases width <;> cases columns <;> revert u <;> decide

theorem selection_exists (p : Source.Parameters) (abi : Whole.InputABI p) :
    Nonempty (Selection p abi) := by
  have hn:=abi.blocks.2.1
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
  have hw : 1≤abi.numWarps := by
    have h:=abi.launchConfig.1
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  rcases hn with h|h|h|h
  · exact ⟨⟨.n16,h.symm,.w1,hw⟩⟩
  · exact ⟨⟨.n32,h.symm,.w1,hw⟩⟩
  · exact ⟨⟨.n64,h.symm,.w1,hw⟩⟩
  · exact ⟨⟨.n128,h.symm,.w1,hw⟩⟩

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

#axiomsClean reduction_coverage
#axiomsClean selection_exists
#axiomsClean packed_bounds
#axiomsClean output_coordinate
#specNonCircular Selection avoiding [Native.Graph.eval,FP.Structural.exec]
end
end NativePlan
end VeriTile.Bench.Repairs.AttentionKernel.Whole002
