// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: salamander_kinematics.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "salamander_kinematics.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace salamander {
namespace msgs {

namespace {

const ::google::protobuf::Descriptor* ModelKinematics_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  ModelKinematics_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_salamander_5fkinematics_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AssignDesc_salamander_5fkinematics_2eproto() {
  protobuf_AddDesc_salamander_5fkinematics_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "salamander_kinematics.proto");
  GOOGLE_CHECK(file != NULL);
  ModelKinematics_descriptor_ = file->message_type(0);
  static const int ModelKinematics_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModelKinematics, links_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModelKinematics, joints_),
  };
  ModelKinematics_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      ModelKinematics_descriptor_,
      ModelKinematics::default_instance_,
      ModelKinematics_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModelKinematics, _has_bits_[0]),
      -1,
      -1,
      sizeof(ModelKinematics),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModelKinematics, _internal_metadata_),
      -1);
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_salamander_5fkinematics_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      ModelKinematics_descriptor_, &ModelKinematics::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_salamander_5fkinematics_2eproto() {
  delete ModelKinematics::default_instance_;
  delete ModelKinematics_reflection_;
}

void protobuf_AddDesc_salamander_5fkinematics_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AddDesc_salamander_5fkinematics_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::salamander::msgs::protobuf_AddDesc_salamander_5flinks_2eproto();
  ::salamander::msgs::protobuf_AddDesc_salamander_5fjoints_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\033salamander_kinematics.proto\022\017salamande"
    "r.msgs\032\026salamander_links.proto\032\027salamand"
    "er_joints.proto\"s\n\017ModelKinematics\022.\n\005li"
    "nks\030\001 \003(\0132\037.salamander.msgs.LinkKinemati"
    "cs\0220\n\006joints\030\002 \003(\0132 .salamander.msgs.Joi"
    "ntKinematics", 212);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "salamander_kinematics.proto", &protobuf_RegisterTypes);
  ModelKinematics::default_instance_ = new ModelKinematics();
  ModelKinematics::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_salamander_5fkinematics_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_salamander_5fkinematics_2eproto {
  StaticDescriptorInitializer_salamander_5fkinematics_2eproto() {
    protobuf_AddDesc_salamander_5fkinematics_2eproto();
  }
} static_descriptor_initializer_salamander_5fkinematics_2eproto_;

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ModelKinematics::kLinksFieldNumber;
const int ModelKinematics::kJointsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

ModelKinematics::ModelKinematics()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:salamander.msgs.ModelKinematics)
}

void ModelKinematics::InitAsDefaultInstance() {
}

ModelKinematics::ModelKinematics(const ModelKinematics& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:salamander.msgs.ModelKinematics)
}

void ModelKinematics::SharedCtor() {
  _cached_size_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

ModelKinematics::~ModelKinematics() {
  // @@protoc_insertion_point(destructor:salamander.msgs.ModelKinematics)
  SharedDtor();
}

void ModelKinematics::SharedDtor() {
  if (this != default_instance_) {
  }
}

void ModelKinematics::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* ModelKinematics::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return ModelKinematics_descriptor_;
}

const ModelKinematics& ModelKinematics::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_salamander_5fkinematics_2eproto();
  return *default_instance_;
}

ModelKinematics* ModelKinematics::default_instance_ = NULL;

ModelKinematics* ModelKinematics::New(::google::protobuf::Arena* arena) const {
  ModelKinematics* n = new ModelKinematics;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void ModelKinematics::Clear() {
// @@protoc_insertion_point(message_clear_start:salamander.msgs.ModelKinematics)
  links_.Clear();
  joints_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  if (_internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->Clear();
  }
}

bool ModelKinematics::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:salamander.msgs.ModelKinematics)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .salamander.msgs.LinkKinematics links = 1;
      case 1: {
        if (tag == 10) {
          DO_(input->IncrementRecursionDepth());
         parse_loop_links:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtualNoRecursionDepth(
                input, add_links()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(10)) goto parse_loop_links;
        if (input->ExpectTag(18)) goto parse_loop_joints;
        input->UnsafeDecrementRecursionDepth();
        break;
      }

      // repeated .salamander.msgs.JointKinematics joints = 2;
      case 2: {
        if (tag == 18) {
          DO_(input->IncrementRecursionDepth());
         parse_loop_joints:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtualNoRecursionDepth(
                input, add_joints()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_loop_joints;
        input->UnsafeDecrementRecursionDepth();
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:salamander.msgs.ModelKinematics)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:salamander.msgs.ModelKinematics)
  return false;
#undef DO_
}

void ModelKinematics::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:salamander.msgs.ModelKinematics)
  // repeated .salamander.msgs.LinkKinematics links = 1;
  for (unsigned int i = 0, n = this->links_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->links(i), output);
  }

  // repeated .salamander.msgs.JointKinematics joints = 2;
  for (unsigned int i = 0, n = this->joints_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->joints(i), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:salamander.msgs.ModelKinematics)
}

::google::protobuf::uint8* ModelKinematics::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:salamander.msgs.ModelKinematics)
  // repeated .salamander.msgs.LinkKinematics links = 1;
  for (unsigned int i = 0, n = this->links_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        1, this->links(i), false, target);
  }

  // repeated .salamander.msgs.JointKinematics joints = 2;
  for (unsigned int i = 0, n = this->joints_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        2, this->joints(i), false, target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:salamander.msgs.ModelKinematics)
  return target;
}

int ModelKinematics::ByteSize() const {
// @@protoc_insertion_point(message_byte_size_start:salamander.msgs.ModelKinematics)
  int total_size = 0;

  // repeated .salamander.msgs.LinkKinematics links = 1;
  total_size += 1 * this->links_size();
  for (int i = 0; i < this->links_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->links(i));
  }

  // repeated .salamander.msgs.JointKinematics joints = 2;
  total_size += 1 * this->joints_size();
  for (int i = 0; i < this->joints_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->joints(i));
  }

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void ModelKinematics::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:salamander.msgs.ModelKinematics)
  if (GOOGLE_PREDICT_FALSE(&from == this)) {
    ::google::protobuf::internal::MergeFromFail(__FILE__, __LINE__);
  }
  const ModelKinematics* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const ModelKinematics>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:salamander.msgs.ModelKinematics)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:salamander.msgs.ModelKinematics)
    MergeFrom(*source);
  }
}

void ModelKinematics::MergeFrom(const ModelKinematics& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:salamander.msgs.ModelKinematics)
  if (GOOGLE_PREDICT_FALSE(&from == this)) {
    ::google::protobuf::internal::MergeFromFail(__FILE__, __LINE__);
  }
  links_.MergeFrom(from.links_);
  joints_.MergeFrom(from.joints_);
  if (from._internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->MergeFrom(from.unknown_fields());
  }
}

void ModelKinematics::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:salamander.msgs.ModelKinematics)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ModelKinematics::CopyFrom(const ModelKinematics& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:salamander.msgs.ModelKinematics)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ModelKinematics::IsInitialized() const {

  if (!::google::protobuf::internal::AllAreInitialized(this->links())) return false;
  if (!::google::protobuf::internal::AllAreInitialized(this->joints())) return false;
  return true;
}

void ModelKinematics::Swap(ModelKinematics* other) {
  if (other == this) return;
  InternalSwap(other);
}
void ModelKinematics::InternalSwap(ModelKinematics* other) {
  links_.UnsafeArenaSwap(&other->links_);
  joints_.UnsafeArenaSwap(&other->joints_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata ModelKinematics::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = ModelKinematics_descriptor_;
  metadata.reflection = ModelKinematics_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// ModelKinematics

// repeated .salamander.msgs.LinkKinematics links = 1;
int ModelKinematics::links_size() const {
  return links_.size();
}
void ModelKinematics::clear_links() {
  links_.Clear();
}
const ::salamander::msgs::LinkKinematics& ModelKinematics::links(int index) const {
  // @@protoc_insertion_point(field_get:salamander.msgs.ModelKinematics.links)
  return links_.Get(index);
}
::salamander::msgs::LinkKinematics* ModelKinematics::mutable_links(int index) {
  // @@protoc_insertion_point(field_mutable:salamander.msgs.ModelKinematics.links)
  return links_.Mutable(index);
}
::salamander::msgs::LinkKinematics* ModelKinematics::add_links() {
  // @@protoc_insertion_point(field_add:salamander.msgs.ModelKinematics.links)
  return links_.Add();
}
::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkKinematics >*
ModelKinematics::mutable_links() {
  // @@protoc_insertion_point(field_mutable_list:salamander.msgs.ModelKinematics.links)
  return &links_;
}
const ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkKinematics >&
ModelKinematics::links() const {
  // @@protoc_insertion_point(field_list:salamander.msgs.ModelKinematics.links)
  return links_;
}

// repeated .salamander.msgs.JointKinematics joints = 2;
int ModelKinematics::joints_size() const {
  return joints_.size();
}
void ModelKinematics::clear_joints() {
  joints_.Clear();
}
const ::salamander::msgs::JointKinematics& ModelKinematics::joints(int index) const {
  // @@protoc_insertion_point(field_get:salamander.msgs.ModelKinematics.joints)
  return joints_.Get(index);
}
::salamander::msgs::JointKinematics* ModelKinematics::mutable_joints(int index) {
  // @@protoc_insertion_point(field_mutable:salamander.msgs.ModelKinematics.joints)
  return joints_.Mutable(index);
}
::salamander::msgs::JointKinematics* ModelKinematics::add_joints() {
  // @@protoc_insertion_point(field_add:salamander.msgs.ModelKinematics.joints)
  return joints_.Add();
}
::google::protobuf::RepeatedPtrField< ::salamander::msgs::JointKinematics >*
ModelKinematics::mutable_joints() {
  // @@protoc_insertion_point(field_mutable_list:salamander.msgs.ModelKinematics.joints)
  return &joints_;
}
const ::google::protobuf::RepeatedPtrField< ::salamander::msgs::JointKinematics >&
ModelKinematics::joints() const {
  // @@protoc_insertion_point(field_list:salamander.msgs.ModelKinematics.joints)
  return joints_;
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace msgs
}  // namespace salamander

// @@protoc_insertion_point(global_scope)
