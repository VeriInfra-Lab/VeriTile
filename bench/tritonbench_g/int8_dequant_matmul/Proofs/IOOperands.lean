import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOGeometry
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOInputReads
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteOperands

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOOperands
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 3000000

theorem loadA (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals base)
    (pid z t : Nat) (rows : 0<s.M) (hz : z<c.S) (ht : t<steps s c)
    (ptr : register base .ptr [c.BM,c.BK] "A"=
      fun i : TileIndex [c.BM,c.BK]=>(("A":RegionName),(row s c pid i.1%s.M)*s.K+z*c.BK+i.2.1.val))
    (rk : register base .int [c.BK] "rk"=fun i=>(z*c.BK+i.1.val:Int))
    (i : Fin c.BM) (l : Fin c.BK) :
    BitVec.ofInt 8 (av s c base t (i,l,PUnit.unit))=
      Mathematical.left s c (IOLayout.data q.ops s vals)
        ⟨row s c pid i%s.M,Nat.mod_lt _ rows⟩ t z l := by
  have addr : (row s c pid i%s.M)*s.K+z*c.BK+l.val+t*(c.BK*c.S)=
      (row s c pid i%s.M)*s.K+sourceK c t z l.val := by simp [sourceK]; omega
  simp only [av,ap,ptr,rk,Prod.fst,Prod.snd]
  rw [addr]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt]
  have read (active : sourceK c t z l.val<s.K) :=
    IOInputReads.a s c hasBias vals base cells ⟨row s c pid i%s.M,Nat.mod_lt _ rows⟩
      ⟨sourceK c t z l.val,active⟩
  by_cases even : s.K%(c.BK*c.S)=0
  · have active:=even_active s c even t z l.val ht hz l.isLt
    simp [even,Mathematical.left,active,IOLayout.data,read active]
  · have mask:=source_mask s c t z l.val ht hz l.isLt
    simp only [even,beq_iff_eq,not_false_eq_true,↓reduceIte]
    by_cases active : sourceK c t z l.val<s.K
    · have h:=mask.mpr active
      simp [h,Mathematical.left,active,IOLayout.data,read active]
    · have h : ¬z*c.BK+l.val<s.K-t*(c.BK*c.S) := fun h=>active (mask.mp h)
      simp [h,Mathematical.left,active]

theorem loadB (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (base : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals base)
    (pid z t : Nat) (columns : 0<s.N) (hz : z<c.S) (ht : t<steps s c)
    (ptr : register base .ptr [c.BK,c.BN] "B"=
      fun i : TileIndex [c.BK,c.BN]=>(("B":RegionName),(z*c.BK+i.1.val)*s.N+col s c pid i.2.1%s.N))
    (rk : register base .int [c.BK] "rk"=fun i=>(z*c.BK+i.1.val:Int))
    (j : Fin c.BN) (l : Fin c.BK) :
    BitVec.ofInt 8 (bv s c base t (l,j,PUnit.unit))=
      Mathematical.right s c (IOLayout.data q.ops s vals)
        ⟨col s c pid j%s.N,Nat.mod_lt _ columns⟩ t z l := by
  have addr : (z*c.BK+l.val)*s.N+col s c pid j%s.N+t*(c.BK*c.S*s.N)=
      sourceK c t z l.val*s.N+col s c pid j%s.N := by simp [sourceK]; ring
  simp only [bv,bp,ptr,rk,Prod.fst,Prod.snd]
  rw [addr]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt]
  have read (active : sourceK c t z l.val<s.K) :=
    IOInputReads.b s c hasBias vals base cells ⟨sourceK c t z l.val,active⟩
      ⟨col s c pid j%s.N,Nat.mod_lt _ columns⟩
  by_cases even : s.K%(c.BK*c.S)=0
  · have active:=even_active s c even t z l.val ht hz l.isLt
    simp [even,Mathematical.right,active,IOLayout.data,read active]
  · have mask:=source_mask s c t z l.val ht hz l.isLt
    simp only [even,beq_iff_eq,not_false_eq_true,↓reduceIte]
    by_cases active : sourceK c t z l.val<s.K
    · have h:=mask.mpr active
      simp [h,Mathematical.right,active,IOLayout.data,read active]
    · have h : ¬z*c.BK+l.val<s.K-t*(c.BK*c.S) := fun h=>active (mask.mp h)
      simp [h,Mathematical.right,active]

theorem source_a (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (hz : initial.pids 1<c.S) (t : Nat) (ht : t<steps s c) (i : Fin c.BM) (l : Fin c.BK) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    BitVec.ofInt 8 (av s c base t (i,l,PUnit.unit))=
      Mathematical.left s c (IOLayout.data q.ops s vals)
        ⟨row s c (initial.pids 0) i%s.M,Nat.mod_lt _ (IOGeometry.rows_pos s c _ valid)⟩
        t (initial.pids 1) l := by
  refine loadA q s c hasBias vals _ ?_ _ _ _ (IOGeometry.rows_pos s c _ valid) hz ht ?_ ?_ i l
  · intro channel lane notC mask
    simpa only [prelude_memory] using cells channel lane notC mask
  · exact prelude_a (floating q) q s c hasBias initial valid
  · exact prelude_rk (floating q) q s c hasBias initial

theorem source_b (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
    (cells : IOInputReads.DataCells s c hasBias vals initial)
    (valid : initial.pids 0<Geometry.count (geometry s c))
    (hz : initial.pids 1<c.S) (t : Nat) (ht : t<steps s c) (j : Fin c.BN) (l : Fin c.BK) :
    let base:=(preludeExecution (floating q) q s c hasBias initial).val
    BitVec.ofInt 8 (bv s c base t (l,j,PUnit.unit))=
      Mathematical.right s c (IOLayout.data q.ops s vals)
        ⟨col s c (initial.pids 0) j%s.N,Nat.mod_lt _ (IOGeometry.columns_pos s c _ valid)⟩
        t (initial.pids 1) l := by
  refine loadB q s c hasBias vals _ ?_ _ _ _ (IOGeometry.columns_pos s c _ valid) hz ht ?_ ?_ j l
  · intro channel lane notC mask
    simpa only [prelude_memory] using cells channel lane notC mask
  · exact prelude_b (floating q) q s c hasBias initial valid
  · exact prelude_rk (floating q) q s c hasBias initial

#axiomsClean loadA
#axiomsClean loadB
#axiomsClean source_a
#axiomsClean source_b
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOOperands
