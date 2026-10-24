import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelTensor
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelNativeModel

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelNativeTensor
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelTensor KernelNativeModel
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem read_raw (mode : Mode) (s : State Word) (region : RegionName) (offset : Nat) :
    KernelValues.read mode (mapState Term.raw s) region offset=Term.raw (KernelValues.read mode s region offset) := by
  have per (mode : Mode) (cell : Cell Word) :
      toFloat (dtype mode) ((mapCell Term.raw cell).read (dtype mode).toTileDType)=
        Term.raw (toFloat (dtype mode) (cell.read (dtype mode).toTileDType)) := by
    cases mode <;> cases cell with
    | mk d v => cases d <;> rfl
  exact per mode (s.mem region offset)

/-- Scalar Tensor expression reference: chronological K tiles and source
operand order, independently indexed in actual input Tensor backing storage. -/
def initialExpression (r : Request) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (n : Nat) : Term :=
  if r.hasBias then .binary (some .fp32) .real .add (.literal (some .fp32) .real 0)
    (.cast (some .fp32) (dtype r.mode) .real (.raw (logical r regions prepared raw 4 n)))
  else .literal (some .fp32) .real 0

def aExpression (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (m cur : Nat) (lane : Fin (configs c).bk) : Term :=
  if cur*(configs c).bk+lane.val<r.k then
    .raw (logical r regions prepared raw 2 (m*r.k+(cur*(configs c).bk+lane.val)))
  else paddedZero algebra r.mode

def bExpression (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (n cur : Nat) (lane : Fin (configs c).bk) : Term :=
  if cur*(configs c).bk+lane.val<r.k then
    .raw (logical r regions prepared raw 3 (n*r.k+(cur*(configs c).bk+lane.val)))
  else paddedZero algebra r.mode

def preExpression (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (m n : Nat) : Term :=
  foldN (fun cur acc => .binary (some .fp32) .real .add acc
    (.cast (some .fp32) .fp32 .real
      (.dot (some .fp32) (dtype r.mode) (dtype r.mode) .fp32 {} (selectedInstruction c r.mode) (configs c).bk
        (aExpression r c regions prepared raw m cur) (bExpression r c regions prepared raw n cur))))
    (ceilDiv r.k (configs c).bk) (initialExpression r regions prepared raw n)

def savedWord (profile : LocalProfile) (policy : CompilerPolicy) (r : Request) (c : Fin 98)
    (regions : Fin 5 → RegionName) (prepared : Prepared r regions) (raw : Nat → Word) (m n : Nat) : Word :=
  castWord .real (dtype r.mode) (evaluate profile policy (preExpression r c regions prepared raw m n))

def resultWord (profile : LocalProfile) (policy : CompilerPolicy) (r : Request) (c : Fin 98)
    (regions : Fin 5 → RegionName) (prepared : Prepared r regions) (raw : Nat → Word) (m n : Nat) : Word :=
  castWord .real (dtype r.mode) (evaluate profile policy
    (KernelValues.activation algebra r.activation (preExpression r c regions prepared raw m n)))

theorem initial_bridge (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (s : State Word) (binding : StorageBound r regions prepared raw s)
    (i : TileIndex [(configs c).bm,(configs c).bn])
    (hn : KernelValues.col (parameters r c regions) s i.2.1<r.n) :
    initial algebra r.mode (parameters r c regions) (mapState Term.raw s) i=
      initialExpression r regions prepared raw (KernelValues.col (parameters r c regions) s i.2.1) := by
  have colEq : KernelValues.col (parameters r c regions) (mapState Term.raw s) i.2.1=
      KernelValues.col (parameters r c regions) s i.2.1 := rfl
  simp only [initial,biasValue,colEq,zero,algebra,initialExpression,
    show (parameters r c regions).hasBias=r.hasBias from rfl,
    show (parameters r c regions).columns=r.n from rfl,
    show (parameters r c regions).bias=(if r.hasBias then regions 4 else regions 2) from rfl,
    Nat.mod_eq_of_lt hn,hn,if_true]
  by_cases flag : r.hasBias=Bool.true
  · simp only [flag,if_true]
    have live : used r 4 := by simp [used,flag]
    have size := numel r regions prepared 4 live
    have valid : KernelValues.col (parameters r c regions) s i.2.1<
        (metadata r.mode (prepared.tensors 4)).numel := by simpa [size,tensorShape] using hn
    rw [read_raw,bound_read r regions prepared raw s binding 4 live _ valid]
  · simp [flag]

theorem a_bridge (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (s : State Word) (binding : StorageBound r regions prepared raw s)
    (i : Fin (configs c).bm) (hm : KernelValues.row (parameters r c regions) s i<r.batch.rows)
    (cur : Nat) (inside : cur<ceilDiv r.k (configs c).bk) (lane : Fin (configs c).bk) :
    aValues algebra r.mode (parameters r c regions) (mapState Term.raw s) cur (i,lane,PUnit.unit)=
      aExpression r c regions prepared raw (KernelValues.row (parameters r c regions) s i) cur lane := by
  have mask := KernelRealArithmetic.mask_equiv (parameters r c regions) cur lane.val
    (configured_positive r c regions) inside lane.isLt
  unfold aValues aExpression
  simp only [mask]
  by_cases activeK : cur*(configs c).bk+lane.val<r.k
  · simp only [show (parameters r c regions).inner=r.k from rfl,
      show (parameters r c regions).blockInner=(configs c).bk from rfl,activeK,if_true]
    have rowEq : KernelValues.row (parameters r c regions) (mapState Term.raw s) i=
      KernelValues.row (parameters r c regions) s i := rfl
    have address : aOffset (parameters r c regions) (mapState Term.raw s) cur (i,lane,PUnit.unit)=
        KernelValues.row (parameters r c regions) s i*r.k+(cur*(configs c).bk+lane.val) := by
      unfold aOffset
      simp only [rowEq,show (parameters r c regions).rows=r.batch.rows from rfl,
        show (parameters r c regions).inputRowStride=max r.k 1 from rfl,
        show (parameters r c regions).inputInnerStride=1 from rfl,
        show (parameters r c regions).blockInner=(configs c).bk from rfl,
        Nat.mod_eq_of_lt hm,max_eq_left (show 1≤r.k by omega),Nat.mul_one]
      ring
    rw [address,read_raw]
    have live : used r 2 := by simp [used]
    have size := numel r regions prepared 2 live
    apply congrArg Term.raw
    apply bound_read r regions prepared raw s binding 2 live
    simpa [size,tensorShape] using matrix_index _ _ _ _ hm activeK
  · simp only [show (parameters r c regions).inner=r.k from rfl,
      show (parameters r c regions).blockInner=(configs c).bk from rfl,activeK,if_false]

theorem b_bridge (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (s : State Word) (binding : StorageBound r regions prepared raw s)
    (i : Fin (configs c).bn) (hn : KernelValues.col (parameters r c regions) s i<r.n)
    (cur : Nat) (inside : cur<ceilDiv r.k (configs c).bk) (lane : Fin (configs c).bk) :
    bValues algebra r.mode (parameters r c regions) (mapState Term.raw s) cur (lane,i,PUnit.unit)=
      bExpression r c regions prepared raw (KernelValues.col (parameters r c regions) s i) cur lane := by
  have mask := KernelRealArithmetic.mask_equiv (parameters r c regions) cur lane.val
    (configured_positive r c regions) inside lane.isLt
  unfold bValues bExpression
  simp only [mask]
  by_cases activeK : cur*(configs c).bk+lane.val<r.k
  · simp only [show (parameters r c regions).inner=r.k from rfl,
      show (parameters r c regions).blockInner=(configs c).bk from rfl,activeK,if_true]
    have colEq : KernelValues.col (parameters r c regions) (mapState Term.raw s) i=
      KernelValues.col (parameters r c regions) s i := rfl
    have address : bOffset (parameters r c regions) (mapState Term.raw s) cur (lane,i,PUnit.unit)=
        KernelValues.col (parameters r c regions) s i*r.k+(cur*(configs c).bk+lane.val) := by
      unfold bOffset
      simp only [colEq,show (parameters r c regions).columns=r.n from rfl,
        show (parameters r c regions).weightColStride=max r.k 1 from rfl,
        show (parameters r c regions).weightInnerStride=1 from rfl,
        show (parameters r c regions).blockInner=(configs c).bk from rfl,
        Nat.mod_eq_of_lt hn,max_eq_left (show 1≤r.k by omega),Nat.mul_one]
      ring
    rw [address,read_raw]
    have live : used r 3 := by simp [used]
    have size := numel r regions prepared 3 live
    apply congrArg Term.raw
    apply bound_read r regions prepared raw s binding 3 live
    simpa [size,tensorShape] using matrix_index _ _ _ _ hn activeK
  · simp only [show (parameters r c regions).inner=r.k from rfl,
      show (parameters r c regions).blockInner=(configs c).bk from rfl,activeK,if_false]

theorem fold_congr_before {β : Type} (f g : Nat → β → β) (z : β) :
    ∀ count, (∀ j, j<count → ∀ a, f j a=g j a) → foldN f count z=foldN g count z
  | 0,_ => rfl
  | n+1,h => by
    rw [foldN,foldN,fold_congr_before f g z n (fun j hj a => h j (by omega) a)]
    exact h n (by omega) _

theorem pre_bridge (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) (prepared : Prepared r regions)
    (raw : Nat → Word) (s : State Word) (binding : StorageBound r regions prepared raw s)
    (i : TileIndex [(configs c).bm,(configs c).bn])
    (hm : KernelValues.row (parameters r c regions) s i.1<r.batch.rows)
    (hn : KernelValues.col (parameters r c regions) s i.2.1<r.n) :
    KernelNativeModel.preExpression c r.mode (parameters r c regions) (mapState Term.raw s) i=
      preExpression r c regions prepared raw (KernelValues.row (parameters r c regions) s i.1)
        (KernelValues.col (parameters r c regions) s i.2.1) := by
  unfold KernelNativeModel.preExpression preExpression
  have initialEq := initial_bridge r c regions prepared raw s binding i hn
  rw [initialEq]
  have perBlock (cur : Nat) (inside : cur<ceilDiv r.k (configs c).bk) :
      (fun lane => aValues algebra r.mode (parameters r c regions) (mapState Term.raw s) cur (i.1,lane,PUnit.unit))=
        aExpression r c regions prepared raw (KernelValues.row (parameters r c regions) s i.1) cur ∧
      (fun lane => bValues algebra r.mode (parameters r c regions) (mapState Term.raw s) cur (lane,i.2.1,PUnit.unit))=
        bExpression r c regions prepared raw (KernelValues.col (parameters r c regions) s i.2.1) cur := by
    exact ⟨funext (a_bridge r c regions prepared raw s binding i.1 hm cur inside),
      funext (b_bridge r c regions prepared raw s binding i.2.1 hn cur inside)⟩
  apply fold_congr_before
  intro cur inside acc
  have blockEq := perBlock cur inside
  rw [blockEq.1,blockEq.2]
  rfl

#axiomsClean pre_bridge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelNativeTensor
