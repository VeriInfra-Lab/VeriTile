import bench.tritonbench_g.fused_recurrent_delta.Proofs.ScalarView
import bench.tritonbench_g.fused_recurrent_delta.GeometrySource
import VeriTile.Triton.Float.Control
import VeriTile.Triton.Float.ProfiledRewrite
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelProgram
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def sourceSHA256 := "41d896ade9b5f5d6a0a3fdc3843f36124c6417d426f590ecb35c024f104f8b53"
inductive Kind where
  | forward | backward
  deriving DecidableEq
def blockV (kind : Kind) (s : Shape) := if kind=.forward then s.FBV else s.BBV
def gridV (kind : Kind) (s : Shape) := if kind=.forward then s.FNV else s.BNV
structure Program (kind : Kind) (s : Shape) where
  head : Nat
  value : Nat
  headInside : head<s.BH
  valueInside : value<gridV kind s

def region : Region → RegionName
  | .q => "q" | .k => "k" | .v => "v" | .beta => "beta" | .h0 => "h0"
  | .o => "o" | .ht => "ht" | .do_ => "do" | .dht => "dht" | .dq => "dq"
  | .dk => "dk" | .dv => "dv" | .dbeta => "dbeta" | .dh0 => "dh0"
  | .retq => "retq" | .retk => "retk" | .retv => "retv" | .retbeta => "retbeta"
def source (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) : ComputeKernel :=
  match kind with
  | .forward => Source.forward "q" "k" "v" "beta" "o" "h0" "ht"
      (s.T*s.K) (s.T*s.V) s.B s.H s.T s.K s.V s.BK s.FBV scale f.initial f.final f.vectorBeta
  | .backward => Source.backward "q" "k" "v" "beta" "dht" "dh0" "do" "dq" "dk" "dv" "dbeta" "h0"
      (s.T*s.K) (s.T*s.V) 1 s.B s.H s.T s.K s.V s.BK s.BBV scale
      f.initial f.vectorBeta f.initialGrad f.terminalGrad
def first (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (source kind s f scale).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩
def second (s : Shape) (f : Flags) (scale : ℝ) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (first .backward s f scale).after "i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩
def loop (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) : ComputeStmt :=
  match kind with
  | .forward => .forRange "_i" 0 s.T 1 (first kind s f scale).body
  | .backward => .forRangeDyn "_i" (.constNat 0) (.constNat s.T) (.constNat 1) (first kind s f scale).body
def recompute (s : Shape) (f : Flags) (scale : ℝ) : ComputeStmt :=
  .forRange "i" 0 s.T 1 (second s f scale).body

theorem source_body (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) :
    (source kind s f scale).surfaceBody=(first kind s f scale).before++[loop kind s f scale]++
      (first kind s f scale).after := by
  cases kind <;> cases hb : f.vectorBeta <;>
    simp only [source,Source.forward,Source.backward,hb,Bool.false_eq_true,ite_false,ite_true,
      first,loop,ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte] <;> rfl
theorem backward_tail (s : Shape) (f : Flags) (scale : ℝ) :
    (first .backward s f scale).after=(second s f scale).before++[recompute s f scale]++
      (second s f scale).after := by
  cases hb : f.vectorBeta <;>
    simp only [source,Source.backward,hb,Bool.false_eq_true,ite_false,ite_true,
      first,second,recompute,ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte] <;> rfl

variable {α : Type} [Inhabited α]
def view (s : Shape) (f : Flags) (u : State α) : Memory α :=
  fun a => (u.mem (region a.region) (Geometry.offset s f a)).read .real

#axiomsClean source_body
#axiomsClean backward_tail
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelProgram
