"""No network/device: compare execution against the immutable current source."""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
import numpy as np
import pytest
from colosseum_policy_server.backends.dreamlabs import DreamLabsBackend, _chunk_from_reply
from colosseum_policy_server.backends.inspect_agent import InspectAgentBackend
from colosseum_policy_server.model_adapters.robots import llm_robot_contract

ROOT = Path(__file__).resolve().parent / 'fixtures'

def baseline(name):
    path = ROOT / 'upstream_backends' / (name + '.py')
    spec = importlib.util.spec_from_file_location('colosseum_policy_server.backends.baseline_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def dream(cls=DreamLabsBackend):
    obj = cls({'checkpoint': 'offline'})
    obj.plan_hz, obj.control_hz, obj.rows_per_call = 10, 30, 3
    obj.plan_s, obj.chunk_plan_s = .05, 0.
    obj.chunk = np.arange(70, dtype=np.float32).reshape(5,14) / 100
    obj.recording_chunk_id = '2'
    obj.recording_evidence = dict(version=1, source='dreamlabs', blocks=[], execution={})
    return obj

def test_dream_interpolation_is_identical_and_retains_fractional_source_rows():
    current, old = dream(), dream(baseline('dreamlabs').DreamLabsBackend)
    np.testing.assert_array_equal(current._targets(), old._targets())
    mapping = current.recording_evidence['execution']
    assert mapping['source_block'] == '2'
    assert mapping['lower_rows'] == [0,0,1]
    np.testing.assert_allclose(mapping['weights'], [.5, 5/6, 1/6])

async def test_dream_reply_preserves_outside_gripper_values_and_unused_tail():
    obj = dream(); obj.replies=1; obj.asked_at=12.; obj.asked_plan_s=0.
    raw = np.zeros((20, 7)); raw[18,6] = 1.2
    reply = dict(q_l_chunk=raw.tolist(),q_r_chunk=raw.tolist())
    future = asyncio.get_running_loop().create_future(); future.set_result(reply); obj.request=future
    obj._install_reply()
    block = obj.recording_evidence['blocks'][0]
    assert block['values'][18][6] == 1.2 and len(block['values']) == 20
    assert obj.chunk[18,6] == 1
    np.testing.assert_array_equal(obj.chunk, baseline('dreamlabs')._chunk_from_reply(reply))

@pytest.fixture
def inspect_types(monkeypatch):
    for name, names in {'embodiment':['EmbodimentInfo'], 'scene':['Scene'],
            'spaces':['ActionSemantics','Box','CameraSpec','ObservationSpace','StateField','StateSpec'],
            'types':['Observation']}.items():
        module=ModuleType('inspect_robots.'+name)
        for item in names: setattr(module,item,NS)
        monkeypatch.setitem(sys.modules,module.__name__,module)

def agent(cls, chunk, monkeypatch, module):
    monkeypatch.setattr(module, '_state', lambda observation,name,shape: np.zeros(shape))
    obj=cls({}); obj.bound=True; obj.control_mode='joint_pos'; obj.eef_mode=False
    obj.robot=llm_robot_contract('yam'); obj.joint_count=12; obj.gripper_count=2
    obj.gripper_indices=[6,13]; obj.open_value=1; obj.sensors=[]
    obj.action_low=obj.joint_low=np.full(14,-3.); obj.action_high=obj.joint_high=np.full(14,3.)
    obj.action_max_step=np.full(14,.05); obj.model=NS(control_hz=30,max_horizon=2)
    obj.agent=NS(act=lambda observation:chunk)
    return obj

@pytest.mark.parametrize('stop',[False,True])
def test_llm_original_and_stop_substitution_preserve_existing_control(stop,inspect_types,monkeypatch):
    import colosseum_policy_server.backends.inspect_agent as current
    old=baseline('inspect_agent')
    raw=np.full((5,14),.01)
    chunk=NS(actions=[NS(data=row.tolist(),meta={}) for row in raw],
             meta={'request_stop':True,'stop_reason':'done'} if stop else {},control_hz=30)
    obs=NS(instruction='offline',control_step=0,sensors=[])
    now=agent(InspectAgentBackend,chunk,monkeypatch,current)
    before=agent(old.InspectAgentBackend,chunk,monkeypatch,old)
    np.testing.assert_array_equal(now._infer(obs),before._infer(obs))
    evidence=now.recording_evidence
    np.testing.assert_array_equal(evidence['blocks'][0]['values'],raw)
    assert evidence['execution']['kind'] == ('hold' if stop else 'prefix')
    if not stop: assert evidence['execution']['rows']==[0,1]
    else:
        now._infer(NS(instruction='offline',control_step=1,sensors=[]))
        assert now.recording_evidence['blocks']==[]  # Held output never relabeled original.

def test_invalid_llm_execution_still_raises(inspect_types,monkeypatch):
    import colosseum_policy_server.backends.inspect_agent as module
    chunk=NS(actions=[NS(data=[1.]*14,meta={})],meta={},control_hz=30)
    obj=agent(InspectAgentBackend,chunk,monkeypatch,module)
    with pytest.raises(ValueError,match='per-step'):
        obj._infer(NS(instruction='offline',control_step=0,sensors=[]))

async def test_terminal_prefetch_is_drained_without_a_new_request():
    obj=dream(); obj.replies=4; obj.asked_at=1.; obj.asked_plan_s=0.
    reply=dict(q_l_chunk=[[0.]*7]*4,q_r_chunk=[[0.]*7]*4)
    future=asyncio.get_running_loop().create_future(); future.set_result(reply); obj.request=future
    evidence=await obj.drain_recording()
    assert obj.request is None and obj.replies==5
    assert evidence['execution']['kind']=='unused_terminal'
    assert evidence['blocks'][0]['id']=='5'
    assert evidence['blocks'][0]['disposition']=='unused_terminal'

async def test_auto_backend_exposes_active_evidence_and_terminal_reply():
    from colosseum_policy_server.backends.auto import AutoBackend
    obj=AutoBackend({}); obj.active=dream(); obj.active.request=None
    assert obj.recording_evidence is obj.active.recording_evidence
    assert (await obj.drain_recording())['execution']['kind']=='unused_terminal'

async def test_multiblock_and_terminal_evidence_cross_real_loopback_websocket(tmp_path):
    import json
    from colosseum_policy_server import colosseum_pb2 as pb
    from colosseum_policy_server.local_runtime import LocalPolicyRuntime, RuntimeConfig
    from colosseum_policy_server.local_protocol import encode_control, decode_control
    from test_yam_backend import model, observation
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    evidence=dict(version=1,source='dreamlabs',blocks=[dict(id='1',values=[[0.]*14]*20)],
        execution=dict(kind='linear_interpolation_after_gripper_clip',source_block='1',lower_rows=[0],upper_rows=[1],weights=[.5]))
    class Backend:
        recording_evidence=evidence
        async def infer(self,*args): return np.zeros((1,14),np.float32)
        async def drain_recording(self): return dict(version=1,source='dreamlabs',blocks=[],execution={'kind':'unused_terminal'})
    item=model(); backend=Backend()
    runtime=LocalPolicyRuntime(RuntimeConfig('127.0.0.1',0,'vla',{},tmp_path,1,{item.name:item}),backend=backend)
    async with serve(runtime.handle,'127.0.0.1',0) as server:
        async with connect(f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}',proxy=None) as ws:
            await ws.send(encode_control(dict(type='prepare',protocol_version=1,run_id='raw',preparation_id='prep',robot_type='yam',model=item.public_spec())))
            assert json.loads(decode_control(await ws.recv())['message'])['roboeval_evidence_v1']
            await ws.send(pb.RelayFrame(protocol_version=1,type=pb.OBSERVATION,session_id='raw',sequence=1,deadline_ms=2000,payload=observation().SerializeToString()).SerializeToString())
            plan=pb.ActionPlan.FromString(pb.RelayFrame.FromString(await ws.recv()).payload)
            assert json.loads(plan.action_evidence_json)==evidence
            await ws.send(pb.RelayFrame(protocol_version=1,type=pb.LOCAL_CONTROL,session_id='raw',payload=b'roboeval_drain_v1').SerializeToString())
            terminal=pb.RelayFrame.FromString(await ws.recv())
            assert json.loads(terminal.payload)['recording_drain']['execution']['kind']=='unused_terminal'
            await ws.send(pb.RelayFrame(protocol_version=1,type=pb.SESSION_CLOSE,session_id='raw').SerializeToString())
