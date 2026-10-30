import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Native
/-! Native FP32 model boundary. Only actual instruction ports are parameters;
the source's +0 and 1 literals have their fixed IEEE encodings. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.NativeWords
open VeriTile.Triton Native
noncomputable section

def zero : Word := BitVec.ofNat 32 0
def one : Word := BitVec.ofNat 32 0x3f800000

structure Instructions where
  neg : Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  neu : Word → Word → Bool
  gt : Word → Word → Bool

def ports (native : Instructions) : Native.Instructions Word where
  literal := fun value => if value = 1 then one else zero
  neg := native.neg
  sub := native.sub
  mul := native.mul
  fma := native.fma
  neu := native.neu
  gt := native.gt
  unused := zero

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.NativeWords
