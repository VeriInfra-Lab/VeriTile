import bench.tritonbench_g.streamk_matmul.Sources.Mathematical_HostDefinitions
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Plan
def select (h : HostInput) : HostPlan :=
  if h.k=0 then ⟨ceilDiv h.m h.bm,ceilDiv h.n h.bn,0,0,
    ceilDiv h.m h.bm*ceilDiv h.n h.bn,0,0⟩ else hostPlan h

def edge (q r p : Nat) : Nat := p*q+min p r

def next (I stop current : Nat) : Nat := min (current+(I-current%I)) stop

theorem next_progress (I stop current : Nat) (hI : 0 < I) (h : current < stop) :
    current < next I stop current ∧ next I stop current ≤ stop := by
  have hm:=Nat.mod_lt current hI
  simp only [next]
  omega


def tiles (h : HostInput) := ceilDiv h.m h.bm*ceilDiv h.n h.bn

theorem selected_zeroK (h : HostInput) (hk : h.k=0) :
    (select h).streamTiles=0 ∧ (select h).blockingTiles=tiles h ∧ (select h).iterations=0 := by
  simp [select,hk,tiles]

theorem selected_no_programs (h : HostInput) (hp : h.programs=0) : (select h).streamTiles=0 := by
  by_cases hk:h.k=0 <;> simp [select,hostPlan,hp,hk]

theorem selected_active (h : HostInput) (hs : 0 < (select h).streamTiles) :
    0 < h.programs ∧ 0 < h.k := by
  constructor
  · by_contra hp
    have hz:h.programs=0:=by omega
    rw [selected_no_programs h hz] at hs
    omega
  · by_contra hk
    have hz:h.k=0:=by omega
    rw [(selected_zeroK h hz).1] at hs
    omega

theorem selected_iterations (h : HostInput) : (select h).iterations=ceilDiv h.k h.bk := by
  by_cases hk:h.k=0
  · simp only [select,hk,if_pos,ceilDiv,Nat.zero_add]
    by_cases hb:h.bk=0
    · simp [hb]
    · exact (Nat.div_eq_of_lt (by omega)).symm
  · simp [select,hk,hostPlan]

theorem iterations_positive (h : HostInput) (hk : 0 < h.k) (hB : 0 < h.bk) :
    0 < (select h).iterations := by
  rw [selected_iterations]
  exact Nat.div_pos (by omega) hB

end Plan
namespace Protocol
open Plan
structure Chunk where
  lo : Nat
  hi : Nat
  deriving DecidableEq, Repr

def Chunk.tile (I : Nat) (c : Chunk) := c.lo/I

def boundary (I current : Nat) := (current/I+1)*I

def chunks (I stop start : Nat) (hI : 0 < I) : List Chunk :=
  if h : start < stop then
    ⟨start,next I stop start⟩ :: chunks I stop (next I stop start) hI
  else []
termination_by stop-start
decreasing_by
  have hn:=next_progress I stop start hI (by assumption)
  omega

def readsScratch (q r I tile program : Nat) : Prop :=
  edge q r program < edge q r (program+1) ∧ edge q r (program+1)%I≠0 ∧
  edge q r (program+1)/I=tile

def programChunks (S I P p : Nat) (hI : 0 < I) : List Chunk :=
  chunks I (edge (S*I/P) (S*I%P) (p+1)) (edge (S*I/P) (S*I%P) p) hI

end Protocol
namespace Geometry
structure Contiguous (rows cols sr sc : Nat) : Prop where
  row : 1 < rows → 0 < cols → sr=cols
  col : 0 < rows → 1 < cols → sc=1

theorem contiguous_offset (rows cols sr sc r c : Nat)
    (h : Contiguous rows cols sr sc) (hr : r < rows) (hc : c < cols) :
    r*sr+c*sc=r*cols+c := by
  have row:r*sr=r*cols:=by
    by_cases hd:1 < rows
    · rw [h.row hd (by omega)]
    · have hz:r=0:=by omega
      simp [hz]
  have col:c*sc=c:=by
    by_cases hd:1 < cols
    · rw [h.col (by omega) hd,Nat.mul_one]
    · have hz:c=0:=by omega
      simp [hz]
  rw [row,col]

theorem packed_bound (rows cols r c : Nat) (hr : r < rows) (hc : c < cols) :
    r*cols+c < rows*cols := by
  have h:=Nat.mul_le_mul_right cols (show r+1 ≤ rows by omega)
  rw [Nat.add_mul,Nat.one_mul] at h
  omega

theorem input_mask_safe (rows cols sr sc r c : Nat)
    (h : Contiguous rows cols sr sc) (hr : r < rows) (hc : c < cols) :
    r*sr+c*sc < rows*cols := by
  rw [contiguous_offset rows cols sr sc r c h hr hc]
  exact packed_bound rows cols r c hr hc

theorem scratch_bounds (P BM BN p r c : Nat) (hp : p < P) (hr : r < BM) (hc : c < BN) :
    p*BM*BN+r*BN+c < P*BM*BN := by
  have hrow:=packed_bound P BM p r hp hr
  have hcell:=packed_bound (P*BM) BN (p*BM+r) c hrow hc
  simpa only [Nat.add_mul,Nat.add_assoc] using hcell

end Geometry
namespace Memory
structure State (α : Type) (A B C S E : Nat) where
  inputA : Fin A → α
  inputB : Fin B → α
  output : Fin C → α
  scratch : Fin S → α
  exterior : Fin E → α


inductive Event (α : Type) (A B C S : Nat) where
  | storeOutput (index : Fin C) (value : (Fin A → α) → (Fin B → α) → α)
  | storeScratch (index : Fin S) (value : (Fin A → α) → (Fin B → α) → α)
  | atomic (output : Fin C) (scratch : Fin S)

def step (add : α → α → α) (e : Event α A B C S) (s : State α A B C S E) : State α A B C S E :=
  match e with
  | .storeOutput i v => {s with output:=Function.update s.output i (v s.inputA s.inputB)}
  | .storeScratch i v => {s with scratch:=Function.update s.scratch i (v s.inputA s.inputB)}
  | .atomic i j => {s with output:=Function.update s.output i (add (s.output i) (s.scratch j))}

def run (add : α → α → α) (events : List (Event α A B C S)) (s : State α A B C S E) :=
  events.foldl (fun current event => step add event current) s

end Memory
end VeriTile.Bench.Repairs.StreamK.Whole001
