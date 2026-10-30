import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteGeometry
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteModel

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 80000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- The source regions are views into original storage identities and offsets.
Read-only aliases are retained by this map. -/
def sourceMemory (q : IntegerModel I F H) (b : Buffers) (m : Memory F H)
    (region : RegionName) (offset : Nat) : Cell (Sum F H) :=
  if region="A" then .mk .int (read8 m (b.a,b.aOffset+offset)).toInt
  else if region="B" then .mk .int (read8 m (b.b,b.bOffset+offset)).toInt
  else if region="state_x" then .mk .real (.inl (read32 q.ops m (b.x,b.xOffset+offset)))
  else if region="state_w" then .mk .real (.inl (read32 q.ops m (b.w,b.wOffset+offset)))
  else if region="bias" then .mk .real (.inl (read32 q.ops m (b.bias,b.biasOffset+offset)))
  else if region="C" then .mk .fp16 (.inr (read16 q.ops m (b.out,b.outOffset+offset)))
  else .mk .int 0

def initialState (q : IntegerModel I F H) [Inhabited (Sum F H)] (b : Buffers)
    (m : Memory F H) (divisor : F) (pid split : Nat) : State (Sum F H) :=
  ({mem:=sourceMemory q b m,regs:=fun _ _ _=>none,
    pids:=fun ax=>if ax=0 then pid else if ax=1 then split else 0,
    numPids:=fun _=>0,undef:=fun dtype _ _=>defaultValue dtype} : State (Sum F H)).setReg
    "divfactor" .fp32 [] (fun _=>.inl divisor)

theorem prelude_memory {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    (preludeExecution M q s c bias initial).val.mem=initial.mem := by simp [preludeExecution]

theorem loadA (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (b : Buffers) (m : Memory F H) (base : State (Sum F H))
    (pid z t : Nat) (hz : z<c.S) (ht : t<steps s c)
    (ptr : register base .ptr [c.BM,c.BK] "A"=
      fun i : TileIndex [c.BM,c.BK]=>(("A" : RegionName),(row s c pid i.1%s.M)*s.K+z*c.BK+i.2.1.val))
    (rk : register base .int [c.BK] "rk"=fun i=>(z*c.BK+i.1.val:Int))
    (memory : base.mem=sourceMemory q b m) (i : Fin c.BM) (l : Fin c.BK) :
    BitVec.ofInt 8 (av s c base t (i,l,PUnit.unit))=
      operandA s c b m (row s c pid i%s.M) t z l := by
  have addr : (row s c pid i%s.M)*s.K+z*c.BK+l.val+t*(c.BK*c.S)=
      (row s c pid i%s.M)*s.K+sourceK c t z l.val := by simp [sourceK];omega
  simp only [av,ap,ptr,rk,memory,Prod.fst,Prod.snd,sourceMemory,ite_true,Cell.read_same]
  rw [addr]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt]
  by_cases even : s.K%(c.BK*c.S)=0
  · have active:=even_active s c even t z l.val ht hz l.isLt
    simp [even,operandA,active,addr,Nat.add_assoc]
  · have mask:=source_mask s c t z l.val ht hz l.isLt
    simp only [even,beq_iff_eq,not_false_eq_true,↓reduceIte]
    by_cases active : sourceK c t z l.val<s.K
    · have h:=mask.mpr active
      simp [h,operandA,active,addr,Nat.add_assoc]
    · have h : ¬z*c.BK+l.val<s.K-t*(c.BK*c.S) := fun h=>active (mask.mp h)
      simp [h,operandA,active]

theorem source_a (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (pid z t : Nat) (valid : pid<Geometry.count (geometry s c))
    (hz : z<c.S) (ht : t<steps s c) (i : Fin c.BM) (l : Fin c.BK) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    BitVec.ofInt 8 (av s c base t (i,l,PUnit.unit))=
      operandA s c b m (row s c pid i%s.M) t z l := by
  have valid' : (initialState q b m divisor pid z).pids 0<Geometry.count (geometry s c) := by
    simpa [initialState] using valid
  refine loadA q s c b m (preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    pid z t hz ht ?_ ?_ ?_ i l
  · simpa [initialState] using prelude_a (floating q) q s c bias (initialState q b m divisor pid z) valid'
  · simpa [initialState] using prelude_rk (floating q) q s c bias (initialState q b m divisor pid z)
  · simpa [initialState] using prelude_memory (floating q) q s c bias (initialState q b m divisor pid z)

#axiomsClean source_a

theorem loadB (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (b : Buffers) (m : Memory F H) (base : State (Sum F H))
    (pid z t : Nat) (hz : z<c.S) (ht : t<steps s c)
    (ptr : register base .ptr [c.BK,c.BN] "B"=
      fun i : TileIndex [c.BK,c.BN]=>(("B" : RegionName),(z*c.BK+i.1.val)*s.N+col s c pid i.2.1%s.N))
    (rk : register base .int [c.BK] "rk"=fun i=>(z*c.BK+i.1.val:Int))
    (memory : base.mem=sourceMemory q b m) (j : Fin c.BN) (l : Fin c.BK) :
    BitVec.ofInt 8 (bv s c base t (l,j,PUnit.unit))=
      operandB s c b m (col s c pid j%s.N) t z l := by
  have addr : (z*c.BK+l.val)*s.N+col s c pid j%s.N+t*(c.BK*c.S*s.N)=
      sourceK c t z l.val*s.N+col s c pid j%s.N := by simp [sourceK];ring
  simp only [bv,bp,ptr,rk,memory,Prod.fst,Prod.snd]
  rw [addr]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt]
  simp only [sourceMemory,show ("B" : RegionName)≠"A" by decide,ite_false,ite_true,Cell.read_same]
  by_cases even : s.K%(c.BK*c.S)=0
  · have active:=even_active s c even t z l.val ht hz l.isLt
    simp [even,operandB,active,Nat.add_assoc]
  · have mask:=source_mask s c t z l.val ht hz l.isLt
    simp only [even,beq_iff_eq,not_false_eq_true,↓reduceIte]
    by_cases active : sourceK c t z l.val<s.K
    · have h:=mask.mpr active
      simp [h,operandB,active,Nat.add_assoc]
    · have h : ¬z*c.BK+l.val<s.K-t*(c.BK*c.S) := fun h=>active (mask.mp h)
      simp [h,operandB,active]

theorem source_b (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (pid z t : Nat) (valid : pid<Geometry.count (geometry s c))
    (hz : z<c.S) (ht : t<steps s c) (j : Fin c.BN) (l : Fin c.BK) :
    let base:=(preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    BitVec.ofInt 8 (bv s c base t (l,j,PUnit.unit))=
      operandB s c b m (col s c pid j%s.N) t z l := by
  have valid' : (initialState q b m divisor pid z).pids 0<Geometry.count (geometry s c) := by
    simpa [initialState] using valid
  refine loadB q s c b m (preludeExecution (floating q) q s c bias (initialState q b m divisor pid z)).val
    pid z t hz ht ?_ ?_ ?_ j l
  · simpa [initialState] using prelude_b (floating q) q s c bias (initialState q b m divisor pid z) valid'
  · simpa [initialState] using prelude_rk (floating q) q s c bias (initialState q b m divisor pid z)
  · simpa [initialState] using prelude_memory (floating q) q s c bias (initialState q b m divisor pid z)

#axiomsClean source_b
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
