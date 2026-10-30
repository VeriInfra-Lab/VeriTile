import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.BmmChunkFwd.K8
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical

abbrev Word := BitVec 32
/-- Fixed native instruction families used by the production configurations.
Each numerical observation is one coordinate of one K8 instruction. -/
inductive Instruction where
  | mma16x8 | wgmma64x32 | wgmma64x64 | wgmma64x128 | wgmma64x256
  deriving DecidableEq, Repr

def instruction (c : Host.Config) : Instruction :=
  if c.warps=2 then .mma16x8 else
  if c.N=32 then .wgmma64x32 else if c.N=64 then .wgmma64x64 else
  if c.N=128 then .wgmma64x128 else .wgmma64x256

/-- Only sixteen input words and one reached carry enter a numerical atom.
The live bit is the native scale-D/carry selector. There is no variable K,
source tile, reduction result, or whole execution in this signature. -/
def Instruction.rows : Instruction → Nat
  | .mma16x8 => 16 | _ => 64
def Instruction.columns : Instruction → Nat
  | .mma16x8 => 8 | .wgmma64x32 => 32 | .wgmma64x64 => 64 | .wgmma64x128 => 128 | .wgmma64x256 => 256
theorem rows_positive (i : Instruction) : 0 < i.rows := by cases i <;> decide
theorem columns_positive (i : Instruction) : 0 < i.columns := by cases i <;> decide
abbrev Coordinate (i : Instruction) := Fin i.rows × Fin i.columns
abbrev Primitive := (i : Instruction) → Coordinate i →
  (Fin 8 → Word) → (Fin 8 → Word) → Word → Bool → Word

def outputRow : (batch : TileShape) → {m n : Nat} → TileIndex (batch++[m,n]) → Fin m
  | [],_,_,i => i.1
  | _::bs,_,_,i => outputRow bs i.2
def outputColumn : (batch : TileShape) → {m n : Nat} → TileIndex (batch++[m,n]) → Fin n
  | [],_,_,i => i.2.1
  | _::bs,_,_,i => outputColumn bs i.2

def coordinate (c : Host.Config) (batch : TileShape) {m n : Nat} (i : TileIndex (batch++[m,n])) :
    Coordinate (instruction c) :=
  (⟨(outputRow batch i).val%(instruction c).rows,Nat.mod_lt _ (rows_positive _)⟩,
   ⟨(outputColumn batch i).val%(instruction c).columns,Nat.mod_lt _ (columns_positive _)⟩)

def leftIndex : (batch : TileShape) → {m k n : Nat} →
    TileIndex (batch++[m,n]) → Fin k → TileIndex (batch++[m,k])
  | [],_,_,_,i,e => (i.1,e,PUnit.unit)
  | _::bs,_,_,_,i,e => (i.1,leftIndex bs i.2 e)
def rightIndex : (batch : TileShape) → {m k n : Nat} →
    TileIndex (batch++[m,n]) → Fin k → TileIndex (batch++[k,n])
  | [],_,_,_,i,e => (e,i.2.1,PUnit.unit)
  | _::bs,_,_,_,i,e => (i.1,rightIndex bs i.2 e)

def left (batch : TileShape) (m k n : Nat) (a : TileIndex (batch++[m,k]) → Word)
    (i : TileIndex (batch++[m,n])) (q : Nat) (e : Fin 8) : Word :=
  if h : q*8+e.val<k then a (leftIndex batch i ⟨q*8+e.val,h⟩) else 0

def right (batch : TileShape) (m k n : Nat) (b : TileIndex (batch++[k,n]) → Word)
    (i : TileIndex (batch++[m,n])) (q : Nat) (e : Fin 8) : Word :=
  if h : q*8+e.val<k then b (rightIndex batch i ⟨q*8+e.val,h⟩) else 0

def liveAt (live : Bool) (q : Nat) : Bool := live || decide (0<q)

def run (mma : Primitive) (inst : Instruction) (coord : Coordinate inst) (a b : Nat → Fin 8 → Word)
    (carry : Word) (live : Bool) : Nat → Word
  | 0 => carry
  | q+1 => mma inst coord (a q) (b q) (run mma inst coord a b carry live q) (liveAt live q)

def plan (mma : Primitive) (c : Host.Config) (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word)
    (carry : Word) (live : Bool) (i : TileIndex (batch++[m,n])) : Word :=
  run mma (instruction c) (coordinate c batch i) (left batch m k n a i) (right batch m k n b i) carry live ((k+7)/8)

/-- The selected lowering uses a live C operand for MMA and scale-D=true for
WGMMA, including the initial literal-zero carry. Saved PTX has an explicit
true predicate; the clear selector is not inferred from floating values. -/
def carryLive (_ : NativeGraph.Expr) : Bool := Bool.true

theorem live_carry (q : Nat) : liveAt Bool.true q=Bool.true := by simp [liveAt]

def Agreements (actual reference : Primitive) (inst : Instruction)
    (coord : Coordinate inst) (a b : Nat → Fin 8 → Word) (carry : Word) (live : Bool) (steps : Nat) : Prop :=
  ∀ q,q<steps →
    actual inst coord (a q) (b q) (run actual inst coord a b carry live q) (liveAt live q)=
      reference inst coord (a q) (b q) (run actual inst coord a b carry live q) (liveAt live q)

theorem transport (actual reference : Primitive) (inst : Instruction)
    (coord : Coordinate inst) (a b : Nat → Fin 8 → Word) (carry : Word) (live : Bool) (steps : Nat)
    (laws : Agreements actual reference inst coord a b carry live steps) :
    run actual inst coord a b carry live steps=run reference inst coord a b carry live steps := by
  induction steps with
  | zero => rfl
  | succ q ih =>
    change actual inst coord (a q) (b q) (run actual inst coord a b carry live q) (liveAt live q)=_
    rw [laws q (by omega),ih (fun i hi => laws i (by omega))]
    rfl

theorem configured_steps (c : Host.Config) (legal : c∈Host.configurations) :
    c.K%8=0 ∧ (c.K+7)/8=c.K/8 ∧ (c.K=32 ∨ c.K=64) ∧
    c.M%(instruction c).rows=0 ∧ c.N%(instruction c).columns=0 := by
  simp only [Host.configurations,List.mem_cons,List.not_mem_nil,or_false] at legal
  rcases legal with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> decide

/-- Every source reduction coordinate has exactly its quotient/remainder K8
instruction and lane. No source K operand is omitted or duplicated. -/
theorem partition (k e : Nat) (inside : e<k) :
    e/8<(k+7)/8 ∧ e%8<8 ∧ e/8*8+e%8=e := by
  have d := Nat.div_add_mod' e 8
  have m := Nat.mod_lt e (by decide : 0<8)
  have ceil := Nat.div_mul_le_self (k+7) 8
  have md := Nat.mod_lt (k+7) (by decide : 0<8)
  have cover := Nat.div_add_mod' (k+7) 8
  refine ⟨?_,m,d⟩
  by_contra h
  have le : (k+7)/8≤e/8 := by omega
  have mul := Nat.mul_le_mul_right 8 le
  omega

theorem unique (q q' : Nat) (e e' : Fin 8) (eq : q*8+e.val=q'*8+e'.val) : q=q' ∧ e=e' := by
  have e1 := e.isLt
  have e2 := e'.isLt
  have qeq : q=q' := by omega
  exact ⟨qeq,Fin.ext (by omega)⟩

/-- A complete configured tile has no synthetic K8 padding lane. -/
theorem lane_inside (c : Host.Config) (legal : c∈Host.configurations)
    (q : Nat) (hq : q<(c.K+7)/8) (e : Fin 8) : q*8+e.val<c.K := by
  have shape := configured_steps c legal
  rw [shape.2.1] at hq
  have div := Nat.div_add_mod' c.K 8
  have mul := Nat.mul_le_mul_right 8 (show q+1≤c.K/8 from hq)
  have bound := e.isLt
  omega

/-- Actual source A[row,K] and B[K,column] selection for a configured step. -/
theorem operands (c : Host.Config) (legal : c∈Host.configurations)
    (a : TileIndex [c.M,c.K] → Word) (b : TileIndex [c.K,c.N] → Word)
    (i : TileIndex [c.M,c.N]) (q : Nat) (hq : q<(c.K+7)/8) (e : Fin 8) :
    left [] c.M c.K c.N a i q e=a (i.1,⟨q*8+e.val,lane_inside c legal q hq e⟩,PUnit.unit) ∧
    right [] c.M c.K c.N b i q e=b (⟨q*8+e.val,lane_inside c legal q hq e⟩,i.2.1,PUnit.unit) := by
  simp [left,right,lane_inside c legal q hq e,leftIndex,rightIndex]

/-- Instruction tile and local coordinate reconstruct each source output. -/
theorem output_coordinate (c : Host.Config) (i : TileIndex [c.M,c.N]) :
    i.1.val/(instruction c).rows*(instruction c).rows+(coordinate c [] i).1.val=i.1.val ∧
    i.2.1.val/(instruction c).columns*(instruction c).columns+(coordinate c [] i).2.val=i.2.1.val := by
  exact ⟨Nat.div_add_mod' _ _,Nat.div_add_mod' _ _⟩

#axiomsClean lane_inside
#axiomsClean operands
#axiomsClean output_coordinate
#axiomsClean transport
#axiomsClean configured_steps
#axiomsClean partition
#axiomsClean unique
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.K8
