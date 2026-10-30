import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelPlan0
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelPlan1
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelPlan3
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelPlan4

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlans
open VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
set_option maxHeartbeats 5000000

def plan_1_1 : PrefixPlan 1 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_1_4 : PrefixPlan 1 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_1_64 : PrefixPlan 1 64 where
  tree := fun _ => NativePlan3.row
  leaves := fun _ => NativePlan3.prefix_correct

def plan_2_1 : PrefixPlan 2 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_2_4 : PrefixPlan 2 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_2_64 : PrefixPlan 2 64 where
  tree := fun _ => NativePlan3.row
  leaves := fun _ => NativePlan3.prefix_correct

def plan_4_1 : PrefixPlan 4 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_4_4 : PrefixPlan 4 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_4_64 : PrefixPlan 4 64 where
  tree := fun h => if h.val < 2 then NativePlan4.row else NativePlan3.row
  leaves := by
    intro h j
    split_ifs
    · exact NativePlan4.prefix_correct j
    · exact NativePlan3.prefix_correct j

def plan_8_1 : PrefixPlan 8 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_8_4 : PrefixPlan 8 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_8_64 : PrefixPlan 8 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_16_1 : PrefixPlan 16 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_16_4 : PrefixPlan 16 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_16_64 : PrefixPlan 16 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_32_1 : PrefixPlan 32 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_32_4 : PrefixPlan 32 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_32_64 : PrefixPlan 32 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

def plan_64_1 : PrefixPlan 64 1 where
  tree := fun _ => NativePlan0.row
  leaves := fun _ => NativePlan0.prefix_correct

def plan_64_4 : PrefixPlan 64 4 where
  tree := fun _ => NativePlan1.row
  leaves := fun _ => NativePlan1.prefix_correct

def plan_64_64 : PrefixPlan 64 64 where
  tree := fun _ => NativePlan4.row
  leaves := fun _ => NativePlan4.prefix_correct

/-- Saved-artifact profile metadata. Every plan is a certified ordered
prefix tree; source/ABI/byte semantics remain in the common kernel proof. -/
structure SavedProfile where
  blockHeads : Nat
  blockChunk : Nat
  bias : Bool
  softplus : Bool
  biasReverse : Bool
  plan : PrefixPlan blockHeads blockChunk
  ptxSHA256 : String

def savedProfiles : List SavedProfile := [
  ⟨1,4,Bool.false,Bool.false,Bool.false,plan_1_4,"9d2e9ff9c9b669c223720704c4ac83bc8e86236d014755ca90201a57d26a551b"⟩,
  ⟨1,4,Bool.false,Bool.true,Bool.false,plan_1_4,"5eec63c1f409525a07e21ce7ff0e3e80c0f2233ba9d00df8d60a6ecfcec4bfcf"⟩,
  ⟨1,4,Bool.true,Bool.false,Bool.false,plan_1_4,"590d2b81d5c329991058f27e4d53a78c7e50b0d374431c63f4766eedfc69b82f"⟩,
  ⟨1,4,Bool.true,Bool.true,Bool.false,plan_1_4,"60f4f2793d6575bfde58356813b9fd58d7df45afa862aa2a16b58610f377abb2"⟩,
  ⟨2,4,Bool.false,Bool.false,Bool.false,plan_2_4,"0b8808fb78209b4d09443aad3473efe9986652e4da2109818a1ea2ea09898e38"⟩,
  ⟨2,4,Bool.false,Bool.true,Bool.false,plan_2_4,"5ae1521cee00ed3503e3c4c8ecf523ded567b084349751a0f28e2fa9a69da037"⟩,
  ⟨2,4,Bool.true,Bool.false,Bool.false,plan_2_4,"6fc1bd304b1d27a8114fa31ccc30f4d2ded007e9f73b2a586048125dad3199ea"⟩,
  ⟨2,4,Bool.true,Bool.true,Bool.false,plan_2_4,"7e7f25536a3477f37ce77a31fea4d9ce52a8fc1271f3630a31cb642ff37ae770"⟩,
  ⟨4,4,Bool.false,Bool.false,Bool.false,plan_4_4,"7fdaf23c147f6cdb97315b78d6bc93e24273fdc3b0aacaf9fe3208d030fb1866"⟩,
  ⟨4,4,Bool.false,Bool.true,Bool.false,plan_4_4,"7a3b938c60f1104a7c77ffc798e71573f83d36e623ad1619ce33d22fc6ea671d"⟩,
  ⟨4,4,Bool.true,Bool.false,Bool.false,plan_4_4,"a396e2d2ed235f601a4d594b4b8c006b62f399f2f38202f25730ec3240254426"⟩,
  ⟨4,4,Bool.true,Bool.true,Bool.false,plan_4_4,"454c0e03be332cb0125b5f40b1cac7e0114b19eb328a97a24c1e195023fc0b8c"⟩,
  ⟨8,4,Bool.false,Bool.false,Bool.false,plan_8_4,"97ded3a4d0f97cba83a88dda7f90ec557155985f846a47440e767c43db7bd172"⟩,
  ⟨8,4,Bool.false,Bool.true,Bool.false,plan_8_4,"878bfec290f84211a53184c46e1af37b129f24e300566da972cd6a65955b128d"⟩,
  ⟨8,4,Bool.true,Bool.false,Bool.false,plan_8_4,"38a1d95fa9607e322882dbd8fb2d388d3344f5972bc80da7ea471debba335d20"⟩,
  ⟨8,4,Bool.true,Bool.true,Bool.false,plan_8_4,"723a7fb1c81fee660ba9c1643ed5d99b5c70fc06fe3adb7cc094202dc69fa5e9"⟩,
  ⟨16,4,Bool.false,Bool.false,Bool.false,plan_16_4,"36d257e538fd259fe423809927759aebb7c263666d5810393b76e4b832783b0f"⟩,
  ⟨16,4,Bool.false,Bool.true,Bool.false,plan_16_4,"bacf6ab555aee7b3c40807de02c4b8ac61daea01720edb6bf2dbad80ddda6151"⟩,
  ⟨16,4,Bool.true,Bool.false,Bool.false,plan_16_4,"b40435e852342809650ea023169df07bee7ad9c0782a1ea75983318ce6f8a8f2"⟩,
  ⟨16,4,Bool.true,Bool.true,Bool.false,plan_16_4,"5b036a9b65e055a2c67e5533ce8ee91aea6d08c5df55e404643e107af9af1bb3"⟩,
  ⟨32,4,Bool.false,Bool.false,Bool.false,plan_32_4,"aa08241023492f16536bd13b35d84aa2f90d6df8db341f834c6180c78d923f0a"⟩,
  ⟨32,4,Bool.false,Bool.true,Bool.false,plan_32_4,"63513bf2c7fe174193c4734d7980bc170708c9c0c26bbe7920a69582e24b60d2"⟩,
  ⟨32,4,Bool.true,Bool.false,Bool.false,plan_32_4,"9ddacbf9dfe689371b6a47caf06f7c9fbf789c523da07e317090f3ddef930e4b"⟩,
  ⟨32,4,Bool.true,Bool.true,Bool.false,plan_32_4,"3199899e2fce543cc97c1e8025aae3dbedec1ca8a1573dc98ed8e3b95c89b22f"⟩,
  ⟨64,4,Bool.false,Bool.false,Bool.false,plan_64_4,"3ffc3c4db82701e2569e87d52c2366daad9b9b03479811a2cb267ffe4fb4567f"⟩,
  ⟨64,4,Bool.false,Bool.true,Bool.false,plan_64_4,"2e7bb8d0eeacd24377fc1099f23c35e73af71ba3f8887e1910146b966794bcd6"⟩,
  ⟨64,4,Bool.true,Bool.false,Bool.false,plan_64_4,"95cc56c85a1a5b9afc3f32061cb2e3ae6e4095cc55a8e59c9ddb81c7822f1dbf"⟩,
  ⟨64,4,Bool.true,Bool.true,Bool.false,plan_64_4,"e2f9ffaaf84916d42747ea21a3a146cd3d185e6c1186dbc6f8b118f5bdc8d037"⟩,
  ⟨1,1,Bool.true,Bool.true,Bool.false,plan_1_1,"ce9a7869d0334bcdab3f70110d3741bb4f357143d9edad16556641c98e034157"⟩,
  ⟨1,64,Bool.true,Bool.true,Bool.false,plan_1_64,"6af2033a7d24791c862c40acf97881a5561029768929c854bf06caedf721d2da"⟩,
  ⟨2,1,Bool.true,Bool.true,Bool.false,plan_2_1,"d194b57446bd0f9f5c1dcc737de6fe8d2442660c0375e2540ee8437fda1dd10f"⟩,
  ⟨2,64,Bool.true,Bool.true,Bool.false,plan_2_64,"e2128a6b7a15d56d1fbfb6786abd24cbf61a7abd308e612b625527a7ea469ebe"⟩,
  ⟨4,1,Bool.true,Bool.true,Bool.false,plan_4_1,"3f52b62bea5e63ace1d45ed9ebb8c48cfd313d5a9508a1803a0088fccb7e6203"⟩,
  ⟨4,64,Bool.true,Bool.true,Bool.false,plan_4_64,"8103e4f821211d4be9757f35fa143a58949db532d9a48501d53af31b92288825"⟩,
  ⟨8,1,Bool.true,Bool.true,Bool.false,plan_8_1,"5734ae801b15b24c1a0b05553ccac2a83d4b44c8a40ecd99018079e32d329283"⟩,
  ⟨8,64,Bool.true,Bool.true,Bool.false,plan_8_64,"7e9d8fcfa2171098f1962cbe7073edec370d20ccbdefde1368d3548d35bc695b"⟩,
  ⟨16,1,Bool.true,Bool.true,Bool.false,plan_16_1,"05b630b5042c1c318d737a2f81d56f517f565be8053bebc5205f21b9afaef27c"⟩,
  ⟨16,64,Bool.true,Bool.true,Bool.false,plan_16_64,"67c203b7ecfac553d6ddb3ea7fac910ba1267802b5b430ed0c6488f3ae5f5893"⟩,
  ⟨32,1,Bool.true,Bool.true,Bool.false,plan_32_1,"cfb04a8f407c1df635d8f5e44c0320588a0c206f471009d6c7e63e3343e09c29"⟩,
  ⟨32,64,Bool.true,Bool.true,Bool.false,plan_32_64,"fa072757acff3a7d4f02d9132cca0055142ad1fae920917e27675300b78b2593"⟩,
  ⟨64,1,Bool.true,Bool.true,Bool.false,plan_64_1,"4f2941a730f8d4335cc7fb2c04bde1fb31c068a8029239eb43d18f88e106c1f4"⟩,
  ⟨64,64,Bool.true,Bool.true,Bool.false,plan_64_64,"f247925b16a5ab61c0c249c1d602c0142ca0f99b1129165c0ee06c86d1acb6fe"⟩
]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.NativePlans
