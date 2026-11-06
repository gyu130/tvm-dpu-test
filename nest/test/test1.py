#"""完整编译链 demo"""
import importlib.util, sys, types, json, numpy as np
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
nest = types.ModuleType('nest'); nest.__path__ = ['nest']; sys.modules['nest'] = nest
load_module('nest.errors', 'nest/errors.py')
load_module('nest.ir.types', 'nest/ir/types.py')
load_module('nest.ir.nodes', 'nest/ir/nodes.py')
load_module('nest.ir.graph', 'nest/ir/graph.py')
load_module('nest.ir.printer', 'nest/ir/printer.py')
load_module('nest.ir', 'nest/ir/__init__.py')
load_module('nest.registry.op_registry', 'nest/registry/op_registry.py')
load_module('nest.registry.standard_ops', 'nest/registry/standard_ops.py')
load_module('nest.registry', 'nest/registry/__init__.py')
load_module('nest.tile.ir', 'nest/tile/ir.py')
load_module('nest.tile.validate', 'nest/tile/validate.py')
load_module('nest.tile.printer', 'nest/tile/printer.py')
load_module('nest.tile.lower', 'nest/tile/lower.py')
load_module('nest.tile.schedule', 'nest/tile/schedule.py')
load_module('nest.tile', 'nest/tile/__init__.py')

from nest.ir import Module, Graph, CallNode, ConstantNode, Value, TensorType, SourceSpan
from nest.tile import schedule_module, lower_program, validate_program, print_program, ScheduleOptions

span = SourceSpan(file='demo.py', node_name='graph', op_name='build')

# --- STEP 1: 构造 Module (Conv+Relu, batch=1) ---
print("="*60, "\nSTEP 1: Module (Conv + Relu, batch=1)\n", "="*60)

x_in = Value(name='x', type=TensorType(dtype='float32', shape=(1,2,5,5), layout='NCHW'))
w_t = TensorType(dtype='float32', shape=(3,2,3,3), layout='OIHW')
w_c = ConstantNode(inputs=[], outputs=[Value(name='w', type=w_t)], span=span,
                  value=np.random.randn(3,2,3,3).astype(np.float32))
# Conv 输出: OH=(5-3)/1+1=3, OW=3
conv_out = Value(name='conv_y', type=TensorType(dtype='float32', shape=(1,3,3,3), layout='NCHW'))
relu_out = Value(name='relu_y', type=TensorType(dtype='float32', shape=(1,3,3,3), layout='NCHW'))

conv_n = CallNode(inputs=['x','w'], outputs=[conv_out], span=span,
    attrs={'pads':[0,0,0,0],'strides':[1,1],'dilations':[1,1],'group':1},
    domain='', op_type='Conv', canonical_name='conv2d')
relu_n = CallNode(inputs=['conv_y'], outputs=[relu_out], span=span,
    attrs={}, domain='', op_type='Relu', canonical_name='relu')

module = Module(name='demo_conv_relu',
    graph=Graph(inputs=[x_in], outputs=[relu_out], nodes=[w_c, conv_n, relu_n]),
    metadata={'frontend':'manual'})
print(f"Module: {module.name}")
for n in module.graph.nodes:
    if isinstance(n, ConstantNode):
        print(f"  CONST {n.output_names} shape={tuple(n.value.shape)}")
    elif isinstance(n, CallNode):
        print(f"  CALL  {n.canonical_name}: {n.inputs} → {n.output_names}")

# --- STEP 2: schedule ---
print("\n" + "="*60, "\nSTEP 2: schedule_module → NestTile Program\n", "="*60)
opts = ScheduleOptions(sram_bytes=1024*1024, alignment=64, conv_oc_tile=3)
prog = schedule_module(module, options=opts)
print(print_program(prog))
validate_program(prog); print("✓ schedule validate OK")

# --- STEP 3: lower → DPU 指令 ---
print("\n" + "="*60, "\nSTEP 3: lower_program → DPU 指令\n", "="*60)
dpu_prog = lower_program(prog)
print(print_program(dpu_prog))

# 统计
def count_items(items, bucket):
    for it in items:
        n = type(it).__name__; bucket[n] = bucket.get(n,0)+1
        if hasattr(it,'body') and it.body: count_items(it.body, bucket)
stats = {}
for st in dpu_prog.stages:
    count_items(st.body, stats); count_items(st.prologue, stats)
    for lp in st.loops:
        if lp.body: count_items(lp.body, stats)
print(f"\n📊 指令统计: {stats}")

# --- STEP 4: validate DPU ---
print("\n" + "="*60, "\nSTEP 4: validate DPU Program\n", "="*60)
validate_program(dpu_prog); print("✓ validate OK")

# --- STEP 5: JSON 导出 ---
print("\n" + "="*60, "\nSTEP 5: JSON 导出\n", "="*60)
payload = {
    'format': 'nest.dpu.v1',
    'source_module': module.name,
    'dpu_program': dpu_prog.to_dict(),
    'stats': stats,
}
out = json.dumps(payload, indent=2, default=str)
path = '/tmp/dpu_program.json'
open(path,'w').write(out)
print(f"✅ {len(out)} 字节 → {path}")

print("\n" + "="*60)
print("🎉 完整编译链: Module → schedule → lower → validate → JSON")
print("="*60)

