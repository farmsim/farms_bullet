// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: salamander_links.proto

#ifndef PROTOBUF_salamander_5flinks_2eproto__INCLUDED
#define PROTOBUF_salamander_5flinks_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
#include "gazebo/msgs/time.pb.h"
#include "gazebo/msgs/pose.pb.h"
#include "gazebo/msgs/vector3d.pb.h"
// @@protoc_insertion_point(includes)

namespace salamander {
namespace msgs {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_salamander_5flinks_2eproto();
void protobuf_AssignDesc_salamander_5flinks_2eproto();
void protobuf_ShutdownFile_salamander_5flinks_2eproto();

class LinkKinematics;
class LinkState;

// ===================================================================

class LinkKinematics : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:salamander.msgs.LinkKinematics) */ {
 public:
  LinkKinematics();
  virtual ~LinkKinematics();

  LinkKinematics(const LinkKinematics& from);

  inline LinkKinematics& operator=(const LinkKinematics& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const LinkKinematics& default_instance();

  void Swap(LinkKinematics* other);

  // implements Message ----------------------------------------------

  inline LinkKinematics* New() const { return New(NULL); }

  LinkKinematics* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const LinkKinematics& from);
  void MergeFrom(const LinkKinematics& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(LinkKinematics* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required string name = 1;
  bool has_name() const;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // repeated .salamander.msgs.LinkState state = 2;
  int state_size() const;
  void clear_state();
  static const int kStateFieldNumber = 2;
  const ::salamander::msgs::LinkState& state(int index) const;
  ::salamander::msgs::LinkState* mutable_state(int index);
  ::salamander::msgs::LinkState* add_state();
  ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkState >*
      mutable_state();
  const ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkState >&
      state() const;

  // @@protoc_insertion_point(class_scope:salamander.msgs.LinkKinematics)
 private:
  inline void set_has_name();
  inline void clear_has_name();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkState > state_;
  friend void  protobuf_AddDesc_salamander_5flinks_2eproto();
  friend void protobuf_AssignDesc_salamander_5flinks_2eproto();
  friend void protobuf_ShutdownFile_salamander_5flinks_2eproto();

  void InitAsDefaultInstance();
  static LinkKinematics* default_instance_;
};
// -------------------------------------------------------------------

class LinkState : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:salamander.msgs.LinkState) */ {
 public:
  LinkState();
  virtual ~LinkState();

  LinkState(const LinkState& from);

  inline LinkState& operator=(const LinkState& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const LinkState& default_instance();

  void Swap(LinkState* other);

  // implements Message ----------------------------------------------

  inline LinkState* New() const { return New(NULL); }

  LinkState* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const LinkState& from);
  void MergeFrom(const LinkState& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(LinkState* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required .gazebo.msgs.Time time = 1;
  bool has_time() const;
  void clear_time();
  static const int kTimeFieldNumber = 1;
  const ::gazebo::msgs::Time& time() const;
  ::gazebo::msgs::Time* mutable_time();
  ::gazebo::msgs::Time* release_time();
  void set_allocated_time(::gazebo::msgs::Time* time);

  // required .gazebo.msgs.Pose pose = 2;
  bool has_pose() const;
  void clear_pose();
  static const int kPoseFieldNumber = 2;
  const ::gazebo::msgs::Pose& pose() const;
  ::gazebo::msgs::Pose* mutable_pose();
  ::gazebo::msgs::Pose* release_pose();
  void set_allocated_pose(::gazebo::msgs::Pose* pose);

  // optional .gazebo.msgs.Vector3d linear_velocity = 3;
  bool has_linear_velocity() const;
  void clear_linear_velocity();
  static const int kLinearVelocityFieldNumber = 3;
  const ::gazebo::msgs::Vector3d& linear_velocity() const;
  ::gazebo::msgs::Vector3d* mutable_linear_velocity();
  ::gazebo::msgs::Vector3d* release_linear_velocity();
  void set_allocated_linear_velocity(::gazebo::msgs::Vector3d* linear_velocity);

  // optional .gazebo.msgs.Vector3d angular_velocity = 4;
  bool has_angular_velocity() const;
  void clear_angular_velocity();
  static const int kAngularVelocityFieldNumber = 4;
  const ::gazebo::msgs::Vector3d& angular_velocity() const;
  ::gazebo::msgs::Vector3d* mutable_angular_velocity();
  ::gazebo::msgs::Vector3d* release_angular_velocity();
  void set_allocated_angular_velocity(::gazebo::msgs::Vector3d* angular_velocity);

  // @@protoc_insertion_point(class_scope:salamander.msgs.LinkState)
 private:
  inline void set_has_time();
  inline void clear_has_time();
  inline void set_has_pose();
  inline void clear_has_pose();
  inline void set_has_linear_velocity();
  inline void clear_has_linear_velocity();
  inline void set_has_angular_velocity();
  inline void clear_has_angular_velocity();

  // helper for ByteSize()
  int RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::gazebo::msgs::Time* time_;
  ::gazebo::msgs::Pose* pose_;
  ::gazebo::msgs::Vector3d* linear_velocity_;
  ::gazebo::msgs::Vector3d* angular_velocity_;
  friend void  protobuf_AddDesc_salamander_5flinks_2eproto();
  friend void protobuf_AssignDesc_salamander_5flinks_2eproto();
  friend void protobuf_ShutdownFile_salamander_5flinks_2eproto();

  void InitAsDefaultInstance();
  static LinkState* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// LinkKinematics

// required string name = 1;
inline bool LinkKinematics::has_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void LinkKinematics::set_has_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void LinkKinematics::clear_has_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void LinkKinematics::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_name();
}
inline const ::std::string& LinkKinematics::name() const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkKinematics.name)
  return name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void LinkKinematics::set_name(const ::std::string& value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:salamander.msgs.LinkKinematics.name)
}
inline void LinkKinematics::set_name(const char* value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:salamander.msgs.LinkKinematics.name)
}
inline void LinkKinematics::set_name(const char* value, size_t size) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:salamander.msgs.LinkKinematics.name)
}
inline ::std::string* LinkKinematics::mutable_name() {
  set_has_name();
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkKinematics.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* LinkKinematics::release_name() {
  // @@protoc_insertion_point(field_release:salamander.msgs.LinkKinematics.name)
  clear_has_name();
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void LinkKinematics::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    set_has_name();
  } else {
    clear_has_name();
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:salamander.msgs.LinkKinematics.name)
}

// repeated .salamander.msgs.LinkState state = 2;
inline int LinkKinematics::state_size() const {
  return state_.size();
}
inline void LinkKinematics::clear_state() {
  state_.Clear();
}
inline const ::salamander::msgs::LinkState& LinkKinematics::state(int index) const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkKinematics.state)
  return state_.Get(index);
}
inline ::salamander::msgs::LinkState* LinkKinematics::mutable_state(int index) {
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkKinematics.state)
  return state_.Mutable(index);
}
inline ::salamander::msgs::LinkState* LinkKinematics::add_state() {
  // @@protoc_insertion_point(field_add:salamander.msgs.LinkKinematics.state)
  return state_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkState >*
LinkKinematics::mutable_state() {
  // @@protoc_insertion_point(field_mutable_list:salamander.msgs.LinkKinematics.state)
  return &state_;
}
inline const ::google::protobuf::RepeatedPtrField< ::salamander::msgs::LinkState >&
LinkKinematics::state() const {
  // @@protoc_insertion_point(field_list:salamander.msgs.LinkKinematics.state)
  return state_;
}

// -------------------------------------------------------------------

// LinkState

// required .gazebo.msgs.Time time = 1;
inline bool LinkState::has_time() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void LinkState::set_has_time() {
  _has_bits_[0] |= 0x00000001u;
}
inline void LinkState::clear_has_time() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void LinkState::clear_time() {
  if (time_ != NULL) time_->::gazebo::msgs::Time::Clear();
  clear_has_time();
}
inline const ::gazebo::msgs::Time& LinkState::time() const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkState.time)
  return time_ != NULL ? *time_ : *default_instance_->time_;
}
inline ::gazebo::msgs::Time* LinkState::mutable_time() {
  set_has_time();
  if (time_ == NULL) {
    time_ = new ::gazebo::msgs::Time;
  }
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkState.time)
  return time_;
}
inline ::gazebo::msgs::Time* LinkState::release_time() {
  // @@protoc_insertion_point(field_release:salamander.msgs.LinkState.time)
  clear_has_time();
  ::gazebo::msgs::Time* temp = time_;
  time_ = NULL;
  return temp;
}
inline void LinkState::set_allocated_time(::gazebo::msgs::Time* time) {
  delete time_;
  time_ = time;
  if (time) {
    set_has_time();
  } else {
    clear_has_time();
  }
  // @@protoc_insertion_point(field_set_allocated:salamander.msgs.LinkState.time)
}

// required .gazebo.msgs.Pose pose = 2;
inline bool LinkState::has_pose() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void LinkState::set_has_pose() {
  _has_bits_[0] |= 0x00000002u;
}
inline void LinkState::clear_has_pose() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void LinkState::clear_pose() {
  if (pose_ != NULL) pose_->::gazebo::msgs::Pose::Clear();
  clear_has_pose();
}
inline const ::gazebo::msgs::Pose& LinkState::pose() const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkState.pose)
  return pose_ != NULL ? *pose_ : *default_instance_->pose_;
}
inline ::gazebo::msgs::Pose* LinkState::mutable_pose() {
  set_has_pose();
  if (pose_ == NULL) {
    pose_ = new ::gazebo::msgs::Pose;
  }
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkState.pose)
  return pose_;
}
inline ::gazebo::msgs::Pose* LinkState::release_pose() {
  // @@protoc_insertion_point(field_release:salamander.msgs.LinkState.pose)
  clear_has_pose();
  ::gazebo::msgs::Pose* temp = pose_;
  pose_ = NULL;
  return temp;
}
inline void LinkState::set_allocated_pose(::gazebo::msgs::Pose* pose) {
  delete pose_;
  pose_ = pose;
  if (pose) {
    set_has_pose();
  } else {
    clear_has_pose();
  }
  // @@protoc_insertion_point(field_set_allocated:salamander.msgs.LinkState.pose)
}

// optional .gazebo.msgs.Vector3d linear_velocity = 3;
inline bool LinkState::has_linear_velocity() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void LinkState::set_has_linear_velocity() {
  _has_bits_[0] |= 0x00000004u;
}
inline void LinkState::clear_has_linear_velocity() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void LinkState::clear_linear_velocity() {
  if (linear_velocity_ != NULL) linear_velocity_->::gazebo::msgs::Vector3d::Clear();
  clear_has_linear_velocity();
}
inline const ::gazebo::msgs::Vector3d& LinkState::linear_velocity() const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkState.linear_velocity)
  return linear_velocity_ != NULL ? *linear_velocity_ : *default_instance_->linear_velocity_;
}
inline ::gazebo::msgs::Vector3d* LinkState::mutable_linear_velocity() {
  set_has_linear_velocity();
  if (linear_velocity_ == NULL) {
    linear_velocity_ = new ::gazebo::msgs::Vector3d;
  }
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkState.linear_velocity)
  return linear_velocity_;
}
inline ::gazebo::msgs::Vector3d* LinkState::release_linear_velocity() {
  // @@protoc_insertion_point(field_release:salamander.msgs.LinkState.linear_velocity)
  clear_has_linear_velocity();
  ::gazebo::msgs::Vector3d* temp = linear_velocity_;
  linear_velocity_ = NULL;
  return temp;
}
inline void LinkState::set_allocated_linear_velocity(::gazebo::msgs::Vector3d* linear_velocity) {
  delete linear_velocity_;
  linear_velocity_ = linear_velocity;
  if (linear_velocity) {
    set_has_linear_velocity();
  } else {
    clear_has_linear_velocity();
  }
  // @@protoc_insertion_point(field_set_allocated:salamander.msgs.LinkState.linear_velocity)
}

// optional .gazebo.msgs.Vector3d angular_velocity = 4;
inline bool LinkState::has_angular_velocity() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void LinkState::set_has_angular_velocity() {
  _has_bits_[0] |= 0x00000008u;
}
inline void LinkState::clear_has_angular_velocity() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void LinkState::clear_angular_velocity() {
  if (angular_velocity_ != NULL) angular_velocity_->::gazebo::msgs::Vector3d::Clear();
  clear_has_angular_velocity();
}
inline const ::gazebo::msgs::Vector3d& LinkState::angular_velocity() const {
  // @@protoc_insertion_point(field_get:salamander.msgs.LinkState.angular_velocity)
  return angular_velocity_ != NULL ? *angular_velocity_ : *default_instance_->angular_velocity_;
}
inline ::gazebo::msgs::Vector3d* LinkState::mutable_angular_velocity() {
  set_has_angular_velocity();
  if (angular_velocity_ == NULL) {
    angular_velocity_ = new ::gazebo::msgs::Vector3d;
  }
  // @@protoc_insertion_point(field_mutable:salamander.msgs.LinkState.angular_velocity)
  return angular_velocity_;
}
inline ::gazebo::msgs::Vector3d* LinkState::release_angular_velocity() {
  // @@protoc_insertion_point(field_release:salamander.msgs.LinkState.angular_velocity)
  clear_has_angular_velocity();
  ::gazebo::msgs::Vector3d* temp = angular_velocity_;
  angular_velocity_ = NULL;
  return temp;
}
inline void LinkState::set_allocated_angular_velocity(::gazebo::msgs::Vector3d* angular_velocity) {
  delete angular_velocity_;
  angular_velocity_ = angular_velocity;
  if (angular_velocity) {
    set_has_angular_velocity();
  } else {
    clear_has_angular_velocity();
  }
  // @@protoc_insertion_point(field_set_allocated:salamander.msgs.LinkState.angular_velocity)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace msgs
}  // namespace salamander

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_salamander_5flinks_2eproto__INCLUDED
