import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Packing
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Software

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPProof
open VeriTile.Triton FP.Structural
open Packing
noncomputable section
set_option maxHeartbeats 8000000

/-- Four-byte storage transport. FP32 loads are exact bit loads; the software
rounder is called separately at every arithmetic node, without contraction. -/
def transport (rest : Algebra Word) (reference : Algebra (FP.Value .fp32)) : Algebra Word :=
  {rest with
    literal:=fun p d x=>Packing.word (reference.literal p d x)
    binary:=fun p d op x y=>Packing.word (reference.binary p d op (Packing.value x) (Packing.value y))
    unary:=fun p op x=>Packing.word (reference.unary p op (Packing.value x))
    fp32Load:=id}

theorem evaluate_transport (rest : Algebra Word) (reference : Algebra (FP.Value .fp32))
    (inputs : Nat → Word) (e : Arithmetic.Expr) :
    Arithmetic.evaluate (transport rest reference) inputs e=
      Packing.word (Arithmetic.evaluate reference (fun i=>Packing.value (inputs i)) e) := by
  induction e with
  | input i=>exact (Packing.word_value (inputs i)).symm
  | literal _=>rfl
  | binary op a b ia ib=>
    simp only [Arithmetic.evaluate,ia,ib]
    simp only [transport,Packing.value_word]
  | unary op a ih=>
    simp only [Arithmetic.evaluate,ih]
    simp only [transport,Packing.value_word]

/-- The domain recursively names only each reached literal, FP32 add/sub/mul
or sign operation. It contains no output, execution, index, frame, protocol,
gradient, or compiler-lowering equality. -/
def ForwardNumerics (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (inputs : Nat → Word) (channel : Fin 11) :=
  Arithmetic.LocalNumerics actual (transport rest (Software.forwardBackend softwareRest cfg))
    inputs (Arithmetic.forwardGraph channel)

def BackwardNumerics (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (inputs : Nat → Word) (component : Fin 3) :=
  Arithmetic.LocalNumerics actual (transport rest (Software.backend softwareRest cfg))
    inputs (Arithmetic.backwardGraph component)

theorem scalar_forward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (inputs : Nat → Word) (channel : Fin 11)
    (numerics : ForwardNumerics actual rest softwareRest cfg inputs channel) :
    Reference.forwardValues actual (inputs 0) (inputs 1) (inputs 2) channel=
      Packing.word (Software.forward cfg (Packing.value (inputs 0)) (Packing.value (inputs 1))
        (Packing.value (inputs 2)) channel) := by
  rw [←Arithmetic.forward_graph actual inputs channel]
  rw [Arithmetic.evaluate_congr actual _ inputs _ numerics,evaluate_transport,Arithmetic.forward_graph]
  exact congrArg Packing.word (congrFun
    (VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.channels_software softwareRest cfg _ _ _) channel)

theorem scalar_backward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (inputs : Nat → Word) (component : Fin 3)
    (numerics : BackwardNumerics actual rest softwareRest cfg inputs component) :
    Reference.backwardValues actual (inputs 0) (inputs 1) (inputs 2)
      (fun c=>inputs (3+c.val)) (fun c=>inputs (14+c.val)) component=
      Packing.word (Software.backward cfg (Packing.value (inputs 0)) (Packing.value (inputs 1))
        (Packing.value (inputs 2)) (fun c=>Packing.value (inputs (3+c.val)))
          (fun c=>Packing.value (inputs (14+c.val))) component) := by
  rw [←Arithmetic.backward_graph actual inputs component]
  rw [Arithmetic.evaluate_congr actual _ inputs _ numerics,evaluate_transport,Arithmetic.backward_graph]
  exact congrArg Packing.word (congrFun (Software.backward_software softwareRest cfg _ _ _ _ _) component)

#axiomsClean evaluate_transport
#axiomsClean scalar_forward
#axiomsClean scalar_backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPProof
