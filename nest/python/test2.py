import importlib.util, sys, types, numpy as np, struct
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod); return mod

nest = types.ModuleType('nest'); nest.__path__=['nest']; sys.modules['nest']=nest
load('nest.errors', 'nest/errors.py')
load('nest.ir.types', 'nest/ir/types.py')
load('nest.ir.nodes', 'nest/ir/nodes.py')
load('nest.ir.graph', 'nest/ir/graph.py')
load('nest.ir.printer', 'nest/ir/printer.py')
load('nest.ir', 'nest/ir/__init__.py')
load('nest.registry.op_registry', 'nest/registry/op_registry.py')
load('nest.registry.standard_ops', 'nest/registry/standard_ops.py')
# 不 import registry/__init__.py — 它会 import nest/__init__.py → frontend → onnx
load('nest.tile.ir', 'nest/tile/ir.py')
load('nest.tile.validate', 'nest/tile/validate.py')
load('nest.tile.printer', 'nest/tile/printer.py')
load('nest.tile.lower', 'nest/tile/lower.py')
load('nest.tile.encoder', 'nest/tile/encoder.py')
load('nest.tile.schedule', 'nest/tile/schedule.py')
# tile/__init__.py 也不 import — 直接 import 各个子模块

from nest.ir import Module, Graph, CallNode, ConstantNode, Value, TensorType, SourceSpan
from nest.tile.schedule import schedule_module, ScheduleOptions
from nest.tile.lower import lower_program
from nest.tile.validate import validate_program
from nest.tile.encoder import encode_items, disassemble
span = SourceSpan(file='demo.py', node_name='graph', op_name='build')

def run_case(name, mod, opts):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    dpu = lower_program(schedule_module(mod, options=opts))
    validate_program(dpu); print("✓ lower+validate")
    all_w = []
    for st in dpu.stages:
        r = encode_items(st.body)
        print(f"--- {st.name} ({len(r)} words) ---")
        for i,(w,d) in enumerate(zip(r.words, r.disasm)):
            print(f"  [{i:3d}] 0x{w:08x}  {d}")
        all_w.extend(r.words)
    bin_path = f'/tmp/dpu_{name}.bin'
    with open(bin_path,'wb') as f:
        for w in all_w: f.write(struct.pack('<I', w))
    with open(bin_path,'rb') as f: br=f.read()
    wb=[struct.unpack('<I', br[i:i+4])[0] for i in range(0,len(br),4)]
    ok = all(a==b for a,b in zip(all_w,wb))
    print(f"✅ {bin_path}: {len(all_w)*4} bytes ({len(all_w)} words) 往返一致={ok}")
    return all_w

# gemm: Y(2,4) = A(2,3) * B(3,4)
a_in = Value(name='A', type=TensorType(dtype='float32', shape=(2,3), layout='linear'))
b_c = ConstantNode(inputs=[], outputs=[Value(name='B', type=TensorType(dtype='float32', shape=(3,4), layout='linear'))],
                  span=span, value=np.random.randn(3,4).astype(np.float32))
y_out = Value(name='Y', type=TensorType(dtype='float32', shape=(2,4), layout='linear'))
gemm_n = CallNode(inputs=['A','B'], outputs=[y_out], span=span,
    attrs={'transA':0,'transB':0}, domain='', op_type='Gemm', canonical_name='gemm')
mod_gemm = Module(name='g', graph=Graph(inputs=[a_in], outputs=[y_out], nodes=[b_c, gemm_n]))
w1 = run_case('gemm', mod_gemm, ScheduleOptions(sram_bytes=1024*1024, gemm_n_tile=4))

# conv2d + relu
x_in = Value(name='x', type=TensorType(dtype='float32', shape=(1,2,5,5), layout='NCHW'))
w_t = TensorType(dtype='float32', shape=(3,2,3,3), layout='OIHW')
w_c2 = ConstantNode(inputs=[], outputs=[Value(name='w', type=w_t)], span=span, value=np.random.randn(3,2,3,3).astype(np.float32))
conv_y = Value(name='conv_y', type=TensorType(dtype='float32', shape=(1,3,3,3), layout='NCHW'))
relu_y = Value(name='relu_y', type=TensorType(dtype='float32', shape=(1,3,3,3), layout='NCHW'))
conv_n = CallNode(inputs=['x','w'], outputs=[conv_y], span=span,
    attrs={'pads':[0,0,0,0],'strides':[1,1],'group':1}, domain='', op_type='Conv', canonical_name='conv2d')
relu_n = CallNode(inputs=['conv_y'], outputs=[relu_y], span=span, attrs={}, domain='', op_type='Relu', canonical_name='relu')
mod_conv = Module(name='c', graph=Graph(inputs=[x_in], outputs=[relu_y], nodes=[w_c2, conv_n, relu_n]))
w2 = run_case('conv2d_relu', mod_conv, ScheduleOptions(sram_bytes=1024*1024, conv_oc_tile=3))

print(f"\n{'='*60}")
print(f"🎉 gemm={len(w1)} words ({len(w1)*4} bytes)")
print(f"   conv2d_relu={len(w2)} words ({len(w2)*4} bytes)")
print("完整 DPU 编译链: Module→schedule→lower→encode→二进制 ✓")
print(f"{'='*60}")
