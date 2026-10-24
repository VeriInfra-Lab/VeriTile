import bench.repairs.tritonbench_g.rotary_transform_ops.KernelSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.RotaryTransformOps.KernelSource

def describe : Nat → ComputeStmt → List String
  | 0,_ => ["..."]
  | _+1,.assign d sh n _ => ["assign "++reprStr n++" : "++reprStr d++" "++reprStr sh]
  | _+1,.store d sh _ _ _ => ["store "++reprStr d++" "++reprStr sh]
  | n+1,.ifThen _ yes => ["if"]++yes.flatMap (describe n)
  | n+1,.ifThenElse _ yes no => ["if/else"]++yes.flatMap (describe n)++["else"]++no.flatMap (describe n)
  | _+1,_ => ["other"]

#eval ((source64_64 "out" "x" "cos" "sin" "cu" "offsets"
  0 1 2 1 2 2 2 1 2 2 2 1 32 8 Bool.false Bool.false Bool.false Bool.false).surfaceBody.flatMap (describe 5))
