import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeParameters
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeRow
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedAffineValues

/-! LayerNorm-specific contracted execution schedule. Loads read the current
opaque tagged memory; registers contain pending expression graphs. Stores
evaluate their actual register operands. Reduction leaves retain their source
owner, including the mean used to center each variance leaf. This is a concrete
native schedule model, not yet a universal compiler-conformance theorem. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

abbrev Vector (α : Type) (block : Nat) := Fin block → Expr α

def vectorStore {α : Type} (s : State α) (region : RegionName) (base N block : Nat)
    (value : Fin block → Cell α) : State α :=
  (TileShape.allIndices [block]).foldl (fun t j =>
    if j.1.val<N then t.write region (base+j.1.val) (value j.1) else t) s

def scalarStore {α : Type} (s : State α) (region : RegionName) (offset : Nat) (value : Cell α) : State α :=
  s.write region offset value

theorem vectorStore_read {α : Type} (s : State α) (region : RegionName) (base N block : Nat)
    (value : Fin block → Cell α) (j : Fin block) (hj : j.val<N) :
    (vectorStore s region base N block value).mem region (base+j.val)=value j :=
  TypedSource.vector_readback region base N block (fun j => value j.1) s (j,PUnit.unit) hj

theorem vectorStore_frame {α : Type} (s : State α) (dest : RegionName) (base N block : Nat)
    (value : Fin block → Cell α) (region : RegionName) (offset : Nat)
    (h : ¬(region=dest ∧ Contract.inRow base N offset)) :
    (vectorStore s dest base N block value).mem region offset=s.mem region offset :=
  TypedSource.vector_frame dest region base N offset block (fun j => value j.1) s h

theorem scalarStore_read {α : Type} (s : State α) (region : RegionName) (offset : Nat) (value : Cell α) :
    (scalarStore s region offset value).mem region offset=value := by simp [scalarStore,State.write]

theorem scalarStore_frame {α : Type} (s : State α) (dest : RegionName) (index : Nat) (value : Cell α)
    (region : RegionName) (offset : Nat) (h : ¬(region=dest ∧ offset=index)) :
    (scalarStore s dest index value).mem region offset=s.mem region offset := by
  simp [scalarStore,State.write,h]

def raw {α : Type} [Inhabited α] (s : State α) (region : RegionName) (base N col : Nat) : Expr α :=
  .cast32 (if col<N then .input ((s.mem region (base+col)).read .real) else .literal 0)

def scalar {α : Type} [Inhabited α] (s : State α) (region : RegionName) (offset : Nat) : Expr α :=
  .cast32 (.input ((s.mem region offset).read .real))

structure StreamRegisters (α : Type) (block : Nat) where
  raw : Vector α block
  scale : Expr α
  value : Vector α block
  keep : Fin block → Bool

def stream {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions)
    (region : RegionName) (base index N : Nat) (p : α) (block : Nat) (s : State α) : StreamRegisters α block :=
  let x := fun j : Fin block => raw s region base N j.val
  let scale := scalar s r.rowscale index
  let seed := (s.mem r.seeds index).read .int
  let random := fun j : Fin block => Expr.random seed j.val
  { raw := x
    scale := scale
    value := fun j =>
      let scaled := if f.rowscale then Expr.mul (x j) scale else x j
      if f.dropout then .selectGT (random j) (.input p)
        (.div scaled (.sub (.literal 1) (.input p))) (.literal 0) else scaled
    keep := fun j => a.greater ((random j).eval a) ((Expr.input p).eval a) }

def storeMask {α : Type} (f : Flags) (r : Regions) (index N block : Nat)
    (values : StreamRegisters α block) (s : State α) : State α :=
  if f.dropout && f.dropoutMask then vectorStore s r.dropoutMask (index*N) N block
    (fun j => .mk .bool (values.keep j)) else s

structure InputRegisters (α : Type) (block : Nat) where
  memory : State α
  value : Vector α block

def inputStage {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) : InputRegisters α block :=
  let first := stream a f r r.x (row*stride.x) row N p block s
  let t := storeMask f r row N block first s
  let second := stream a f r r.x1 (row*stride.x1) (M+row) N p block t
  let u := if f.x1 then storeMask f r (M+row) N block second t else t
  let both := fun j => if f.x1 then Expr.add (first.value j) (second.value j) else first.value j
  let combined := fun j => if f.residual then
      let res := raw u r.residual (row*stride.residual) N j.val
      if f.rowscale && !f.dropout && !f.x1 then Expr.fma (first.raw j) first.scale res
      else .add (both j) res
    else both j
  let final := if f.residualOut then vectorStore u r.residualOut (row*stride.residualOut) N block
    (fun j => .mk .real (a.cast32 ((combined j).eval a))) else u
  ⟨final,combined⟩

structure NormalRegisters (α : Type) (block : Nat) where
  memory : State α
  value : Vector α block
  mean : Nat → Expr α
  centered : Vector α block
  rstd : Nat → Expr α

def normalizationStage {α : Type} (a : Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (r : Regions) (N row : Nat) (eps : α) (hblock : 0<block)
    (input : InputRegisters α block) : NormalRegisters α block :=
  let zero : Fin block := ⟨0,hblock⟩
  let mean := fun owner => if N=1 then input.value zero else
    Expr.div (sumExpr (red owner).mean input.value) (.literal N)
  let s := if f.rms then input.memory else scalarStore input.memory r.mean row
    (.mk .real (a.cast32 ((mean 0).eval a)))
  let centered := fun j : Fin block => if f.rms then input.value j else .sub (input.value j) (mean j.val)
  let masked := fun j : Fin block => if j.val<N then centered j else Expr.literal 0
  let variance := fun owner => Expr.div (squareExpr (red owner).variance masked) (.literal N)
  let rstd := fun owner => Expr.div (.literal 1) (.sqrt (if N=1 then
    .fma (centered zero) (centered zero) (.input eps) else .add (.input eps) (variance owner)))
  let t := scalarStore s r.rstd row (.mk .real (a.cast32 ((rstd 0).eval a)))
  ⟨t,input.value,mean,centered,rstd⟩

def affineStage {α : Type} [Inhabited α] (a : Ops α) (plan plan1 : AffinePlan)
    (f : Flags) (r : Regions) (stride : Strides) (N row : Nat) (input : NormalRegisters α block) : State α :=
  let normalized := fun j : Fin block => if f.rms then
    Expr.mul (input.rstd j.val) (input.centered j) else .mul (input.centered j) (input.rstd j.val)
  let w := fun j : Fin block => raw input.memory r.weight 0 N j.val
  let b := fun j : Fin block => raw input.memory r.bias 0 N j.val
  let y := fun j => affine plan f.bias (normalized j) (w j) (b j)
  let s := vectorStore input.memory r.y (row*stride.y) N block (fun j => .mk .real ((y j).eval a))
  if f.w1 then
    let w1 := fun j : Fin block => raw s r.weight1 0 N j.val
    let b1 := fun j : Fin block => raw s r.bias1 0 N j.val
    let y1 := fun j => affine plan1 f.b1 (normalized j) (w1 j) (b1 j)
    vectorStore s r.y1 (row*stride.y1) N block (fun j => .mk .real ((y1 j).eval a))
  else s

def execute {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (hblock : 0<block) (s : State α) : State α :=
  affineStage a plan plan1 f r stride N row
    (normalizationStage a red f r N row eps hblock (inputStage a f r stride M N row p block s))

/-- Only snapshot of input memory: no output equation or target computation is
fed to `execute`. This independently parameterizes the ordered native target. -/
def snapshot {α : Type} [Inhabited α] (r : Regions) (stride : Strides) (s : State α) : NativeRow.Inputs α where
  x := fun i j => (s.mem r.x (i*stride.x+j)).read .real
  x1 := fun i j => (s.mem r.x1 (i*stride.x1+j)).read .real
  residual := fun i j => (s.mem r.residual (i*stride.residual+j)).read .real
  weight := fun j => (s.mem r.weight j).read .real
  bias := fun j => (s.mem r.bias j).read .real
  weight1 := fun j => (s.mem r.weight1 j).read .real
  bias1 := fun j => (s.mem r.bias1 j).read .real
  rowscale := fun i => (s.mem r.rowscale i).read .real
  seeds := fun i => (s.mem r.seeds i).read .int

#axiomsClean vectorStore_read
#axiomsClean vectorStore_frame
#axiomsClean scalarStore_read
#axiomsClean scalarStore_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
