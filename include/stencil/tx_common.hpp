#pragma once

#include "stencil/dim3.hpp"

#include <cassert>
#include <climits>
#include <iostream>

/* construct a payload that can fit in an MPI tag
 */
inline uint16_t ipc_tag_payload(uint8_t a, uint8_t b) noexcept { return (uint16_t(a) << 8) | uint16_t(b); }

class Message {
private:
  Dim3 ext_; // for sorting by size, otherwise unused

public:
  Dim3 dir_;
  int srcGPU_;
  int dstGPU_;
  Message(Dim3 dir, int srcGPU, int dstGPU) : Message(dir, srcGPU, dstGPU, Dim3(0, 0, 0)) {}
  Message(Dim3 dir, int srcGPU, int dstGPU, Dim3 ext) : ext_(ext), dir_(dir), srcGPU_(srcGPU), dstGPU_(dstGPU) {}

  // true if lhs is larger than rhs. Tie by direction
  static bool by_size(const Message &lhs, const Message &rhs) noexcept {
    if (lhs.ext_.flatten() > rhs.ext_.flatten()) {
      return true;
    } else if (lhs.ext_.flatten() < rhs.ext_.flatten()) {
      return false;
    } else {
      return lhs < rhs;
    }
  }

  // order by direction
  bool operator<(const Message &rhs) const noexcept { return dir_ < rhs.dir_; }
  bool operator==(const Message &rhs) const noexcept {
    return dir_ == rhs.dir_ && srcGPU_ == rhs.srcGPU_ && dstGPU_ == rhs.dstGPU_;
  }
};

// three bits for the message kind
enum class MsgKind {
  ColocatedEvt = 0,
  ColocatedCurrMem = 1,
  ColocatedNextMem = 2,
  ColocatedBuf = 3,
  ColocatedDev = 4,
  ColocatedNotify = 5,
  ColocatedPtr = 6,
  Other = 7,
};

/*!
  Construct an MPI tag.
  We have observed systems where the max tag value is 1 << 23, so we only use
  23 bits here.
*/
template <MsgKind kind> inline int make_tag(const int payload, const Dim3 dir = Dim3(0, 0, 0)) {
  // bit 31 is 0
  int ret = 0;

  // bits 0-2 encode message kind
  int kindInt = static_cast<int>(kind);
  assert(kindInt <= 0b111);
  assert(kindInt >= 0b000);
  ret |= kindInt;

  // bits 3-8 encode direction vector
  assert(dir.x >= -1 && dir.x <= 1);
  assert(dir.y >= -1 && dir.y <= 1);
  assert(dir.z >= -1 && dir.z <= 1);
  int dirBits = 0;
  dirBits |= dir.x == 0 ? 0b00 : (dir.x == 1 ? 0b01 : 0b10);
  dirBits |= (dir.y == 0 ? 0b00 : (dir.y == 1 ? 0b01 : 0b10)) << 2;
  dirBits |= (dir.z == 0 ? 0b00 : (dir.z == 1 ? 0b01 : 0b10)) << 4;
  ret |= (dirBits << 3);

  // bits 9-23 are the payload
  assert(payload < (1 << 15));
  ret |= (payload & 0x7FFF) << 9;

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

  static_assert(DIR_BITS + GPU_BITS + IDX_BITS < sizeof(int) * CHAR_BIT, "too many bits");
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
      preparation may involve coordination with recver, so allow for two phases
   */
  virtual void start_prepare(const std::vector<Message> &outbox) = 0;
  virtual void finish_prepare() = 0;

  /*! initiate an async send
   */
  virtual void send() = 0;

  /*! true if there are states left to complete
   */
  virtual bool active() = 0;

  /*! valid to call next() to continue with the send
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
  /*! prepare reciever to recv these messages
   */
  virtual void start_prepare(const std::vector<Message> &inbox) = 0;
  virtual void finish_prepare() = 0;

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
