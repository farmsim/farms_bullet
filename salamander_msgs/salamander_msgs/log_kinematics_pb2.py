# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: log_kinematics.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import time_pb2 as time__pb2
from . import pose_pb2 as pose__pb2
from . import joint_pb2 as joint__pb2
from . import vector3d_pb2 as vector3d__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='log_kinematics.proto',
  package='salamander.msgs',
  syntax='proto2',
  serialized_pb=_b('\n\x14log_kinematics.proto\x12\x0fsalamander.msgs\x1a\ntime.proto\x1a\npose.proto\x1a\x0bjoint.proto\x1a\x0evector3d.proto\"t\n\x0fModelKinematics\x12.\n\x05links\x18\x01 \x03(\x0b\x32\x1f.salamander.msgs.LinkKinematics\x12\x31\n\x06joints\x18\x02 \x03(\x0b\x32!.salamander.msgs.JointsKinematics\"I\n\x0eLinkKinematics\x12\x0c\n\x04name\x18\x01 \x02(\t\x12)\n\x05state\x18\x02 \x03(\x0b\x32\x1a.salamander.msgs.LinkState\"\xae\x01\n\tLinkState\x12\x1f\n\x04time\x18\x01 \x02(\x0b\x32\x11.gazebo.msgs.Time\x12\x1f\n\x04pose\x18\x02 \x02(\x0b\x32\x11.gazebo.msgs.Pose\x12.\n\x0flinear_velocity\x18\x03 \x01(\x0b\x32\x15.gazebo.msgs.Vector3d\x12/\n\x10\x61ngular_velocity\x18\x04 \x01(\x0b\x32\x15.gazebo.msgs.Vector3d\"C\n\x10JointsKinematics\x12\x0c\n\x04name\x18\x01 \x02(\t\x12!\n\x05joint\x18\x02 \x03(\x0b\x32\x12.gazebo.msgs.Joint')
  ,
  dependencies=[time__pb2.DESCRIPTOR,pose__pb2.DESCRIPTOR,joint__pb2.DESCRIPTOR,vector3d__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_MODELKINEMATICS = _descriptor.Descriptor(
  name='ModelKinematics',
  full_name='salamander.msgs.ModelKinematics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='links', full_name='salamander.msgs.ModelKinematics.links', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='joints', full_name='salamander.msgs.ModelKinematics.joints', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=94,
  serialized_end=210,
)


_LINKKINEMATICS = _descriptor.Descriptor(
  name='LinkKinematics',
  full_name='salamander.msgs.LinkKinematics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='salamander.msgs.LinkKinematics.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='state', full_name='salamander.msgs.LinkKinematics.state', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=285,
)


_LINKSTATE = _descriptor.Descriptor(
  name='LinkState',
  full_name='salamander.msgs.LinkState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='time', full_name='salamander.msgs.LinkState.time', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pose', full_name='salamander.msgs.LinkState.pose', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='linear_velocity', full_name='salamander.msgs.LinkState.linear_velocity', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='angular_velocity', full_name='salamander.msgs.LinkState.angular_velocity', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=288,
  serialized_end=462,
)


_JOINTSKINEMATICS = _descriptor.Descriptor(
  name='JointsKinematics',
  full_name='salamander.msgs.JointsKinematics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='salamander.msgs.JointsKinematics.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='joint', full_name='salamander.msgs.JointsKinematics.joint', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=464,
  serialized_end=531,
)

_MODELKINEMATICS.fields_by_name['links'].message_type = _LINKKINEMATICS
_MODELKINEMATICS.fields_by_name['joints'].message_type = _JOINTSKINEMATICS
_LINKKINEMATICS.fields_by_name['state'].message_type = _LINKSTATE
_LINKSTATE.fields_by_name['time'].message_type = time__pb2._TIME
_LINKSTATE.fields_by_name['pose'].message_type = pose__pb2._POSE
_LINKSTATE.fields_by_name['linear_velocity'].message_type = vector3d__pb2._VECTOR3D
_LINKSTATE.fields_by_name['angular_velocity'].message_type = vector3d__pb2._VECTOR3D
_JOINTSKINEMATICS.fields_by_name['joint'].message_type = joint__pb2._JOINT
DESCRIPTOR.message_types_by_name['ModelKinematics'] = _MODELKINEMATICS
DESCRIPTOR.message_types_by_name['LinkKinematics'] = _LINKKINEMATICS
DESCRIPTOR.message_types_by_name['LinkState'] = _LINKSTATE
DESCRIPTOR.message_types_by_name['JointsKinematics'] = _JOINTSKINEMATICS

ModelKinematics = _reflection.GeneratedProtocolMessageType('ModelKinematics', (_message.Message,), dict(
  DESCRIPTOR = _MODELKINEMATICS,
  __module__ = 'log_kinematics_pb2'
  # @@protoc_insertion_point(class_scope:salamander.msgs.ModelKinematics)
  ))
_sym_db.RegisterMessage(ModelKinematics)

LinkKinematics = _reflection.GeneratedProtocolMessageType('LinkKinematics', (_message.Message,), dict(
  DESCRIPTOR = _LINKKINEMATICS,
  __module__ = 'log_kinematics_pb2'
  # @@protoc_insertion_point(class_scope:salamander.msgs.LinkKinematics)
  ))
_sym_db.RegisterMessage(LinkKinematics)

LinkState = _reflection.GeneratedProtocolMessageType('LinkState', (_message.Message,), dict(
  DESCRIPTOR = _LINKSTATE,
  __module__ = 'log_kinematics_pb2'
  # @@protoc_insertion_point(class_scope:salamander.msgs.LinkState)
  ))
_sym_db.RegisterMessage(LinkState)

JointsKinematics = _reflection.GeneratedProtocolMessageType('JointsKinematics', (_message.Message,), dict(
  DESCRIPTOR = _JOINTSKINEMATICS,
  __module__ = 'log_kinematics_pb2'
  # @@protoc_insertion_point(class_scope:salamander.msgs.JointsKinematics)
  ))
_sym_db.RegisterMessage(JointsKinematics)


# @@protoc_insertion_point(module_scope)