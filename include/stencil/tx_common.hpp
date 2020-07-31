#pragma once

#include "stencil/dim3.hpp"

#include <climits>
#include <iostream>

class Message {
private:
public:
  Dim3 dir_;
  int srcGPU_;
  int dstGPU_;
  Message(Dim3 dir, int srcGPU, int dstGPU)
      : dir_(dir), srcGPU_(srcGPU), dstGPU_(dstGPU) {}

  bool operator<(const Message &rhs) const noexcept { return dir_ < rhs.dir_; }
  bool operator==(const Message &rhs) const noexcept {
    return dir_ == rhs.dir_ && srcGPU_ == rhs.srcGPU_ && dstGPU_ == rhs.dstGPU_;
  }
};

enum class MsgKind {
  ColocatedEvt = 0,
  ColocatedMem = 1,
  ColocatedDev = 2,
  ColocatedNotify = 3,
  Other = 4,
};

/*!
  Construct an MPI tag.
  We have observed systems where the max tag value is 1 << 23, so we only use
  23 bits here.
*/
template <MsgKind kind>
inline int make_tag(const int payload, const Dim3 dir = Dim3(0, 0, 0)) {
  int ret = 0;

  // bit 31 is 0

  // bits 0-1 encode message kind
  int kindInt = static_cast<int>(kind);
  assert(kindInt <= 0b11);
  assert(kindInt >= 0b00);
  ret |= kindInt;

  // bits 2-7 encode direction vector
  assert(dir.x >= -1 && dir.x <= 1);
  assert(dir.y >= -1 && dir.y <= 1);
  assert(dir.z >= -1 && dir.z <= 1);
  int dirBits = 0;
  dirBits |= dir.x == 0 ? 0b00 : (dir.x == 1 ? 0b01 : 0b10);
  dirBits |= (dir.y == 0 ? 0b00 : (dir.y == 1 ? 0b01 : 0b10)) << 2;
  dirBits |= (dir.z == 0 ? 0b00 : (dir.z == 1 ? 0b01 : 0b10)) << 4;
  ret |= (dirBits << 2);

  // bits 8-23 are the payload
  assert(payload < (1 << 16));
  ret |= (payload & 0xFFFF) << 8;

  assert(ret >= 0);
  return ret;
}

/*!
  Construct an MPI tag from a gpu id, a direction vector, and a stencil data
  field index
  tags must be non-negative, so MSB must be 0, leaving 31 bits

  data index in buts 0-15     (16 bits)
  gpu id in bits 16-23        ( 8 bits)
  direction vec in bits 24-30 ( 7 bits)
   0 -> 0b00
   1 -> 0b01
  -1 -> 0b10

*/
inline int make_tag(int gpu, int idx, Dim3 dir) {
  static_assert(sizeof(int) == 4, "int is the wrong size");
  constexpr int IDX_BITS = 16;
  constexpr int GPU_BITS = 8;
  constexpr int DIR_BITS = 7;

  static_assert(DIR_BITS + GPU_BITS + IDX_BITS < sizeof(int) * CHAR_BIT,
                "too many bits");
  static_assert(DIR_BITS >= 6, "not enough bits");
  assert(gpu < (1 << GPU_BITS));
  assert(idx < (1 << IDX_BITS));
  assert(dir.x >= -1 && dir.x <= 1);
  assert(dir.y >= -1 && dir.y <= 1);
  assert(dir.z >= -1 && dir.z <= 1);

  int t = 0;

  int idxBits = idx & ((1 << IDX_BITS) - 1);
  int gpuBits = (gpu & ((1 << GPU_BITS) - 1));
  int dirBits = 0;
  dirBits |= dir.x == 0 ? 0b00 : (dir.x == 1 ? 0b01 : 0b10);
  dirBits |= (dir.y == 0 ? 0b00 : (dir.y == 1 ? 0b01 : 0b10)) << 2;
  dirBits |= (dir.z == 0 ? 0b00 : (dir.z == 1 ? 0b01 : 0b10)) << 4;

  t |= idxBits;
  t |= gpuBits << IDX_BITS;
  t |= dirBits << (IDX_BITS + GPU_BITS);

  assert(t >= 0 && "tag must be non-negative");

  return t;
}

/*! a sender that has multiple phases
    sender->send();
    while(sender->active()) {
      if (sender->state_done()) sender->next();
    }
    sender->wait();
*/
class StatefulSender {
public:
  /*! prepare sender to send these messages
   */
  virtual void prepare(std::vector<Message> &outbox) = 0;

  /*! start a send
   */
  virtual void send() = 0;

  /*! true if there are states left to complete
   */
  virtual bool active() = 0;

  /*! call next() to continue with the send
   */
  virtual bool next_ready() = 0;

  /*! move the sender to the next state
   */
  virtual void next() = 0;

  /*! block until the final state is done
      call after sender->active() becomes false
  */
  virtual void wait() = 0;

  virtual ~StatefulSender() {}
};

class StatefulRecver {
public:
  /*! prepare reciever to send these messages
   */
  virtual void prepare(std::vector<Message> &outbox) = 0;

  /*! start a recv
   */
  virtual void recv() = 0;

  /*! true if there are states left to complete
   */
  virtual bool active() = 0;

  /*! call next() to continue with the send
   */
  virtual bool next_ready() = 0;

  /*! move the sender to the next state
   */
  virtual void next() = 0;

  /*! block until the final state is done
      call after sender->active() becomes false
  */
  virtual void wait() = 0;

  virtual ~StatefulRecver() {}
};
