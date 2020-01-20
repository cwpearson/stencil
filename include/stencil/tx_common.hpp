#pragma once

#include "stencil/dim3.hpp"

#include <climits>

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
static int make_tag(int gpu, int idx, Dim3 dir) {
  static_assert(sizeof(int) == 4);
  constexpr int IDX_BITS = 16;
  constexpr int GPU_BITS = 8;
  constexpr int DIR_BITS = 7;

  static_assert(DIR_BITS + GPU_BITS + IDX_BITS < sizeof(int) * CHAR_BIT);
  static_assert(DIR_BITS >= 6);
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

/*!
  Construct an MPI tag from a gpu id and a direction vector
  tags must be non-negative, so MSB must be 0, leaving 31 bits


  gpu id in bits 0-7        ( 8 bits)
  direction vec in bits 8-13 ( 6 bits)
   0 -> 0b00
   1 -> 0b01
  -1 -> 0b10

*/
static int make_tag(int gpu, Dim3 dir) {
  static_assert(sizeof(int) == 4);
  constexpr int GPU_BITS = 8;
  constexpr int DIR_BITS = 6;

  static_assert(DIR_BITS + GPU_BITS < sizeof(int) * CHAR_BIT);
  static_assert(DIR_BITS >= 6);
  assert(gpu < (1 << GPU_BITS));
  assert(dir.x >= -1 && dir.x <= 1);
  assert(dir.y >= -1 && dir.y <= 1);
  assert(dir.z >= -1 && dir.z <= 1);

  int t = 0;

  int gpuBits = (gpu & ((1 << GPU_BITS) - 1));
  int dirBits = 0;
  dirBits |= dir.x == 0 ? 0b00 : (dir.x == 1 ? 0b01 : 0b10);
  dirBits |= (dir.y == 0 ? 0b00 : (dir.y == 1 ? 0b01 : 0b10)) << 2;
  dirBits |= (dir.z == 0 ? 0b00 : (dir.z == 1 ? 0b01 : 0b10)) << 4;

  t |= gpuBits;
  t |= dirBits << GPU_BITS;

  assert(t >= 0 && "tag must be non-negative");

  return t;
}


/*! An asynchronous sender, to be paired with a Recver
 */
class Sender {

public:
  /*! prepare to send n bytes
   */
  virtual void resize(const size_t n) = 0;

  // send n bytes
  virtual void send(const void *data) = 0;

  /*! block until send is complete
   */
  virtual void wait() = 0;

  virtual ~Sender() {}
};

class Recver {
public:
  /*! prepare to recv n bytes
   */
  virtual void resize(const size_t n) = 0;

  // recieve into data
  virtual void recv(void *data) = 0;

  /*! block until recv is complete
   */
  virtual void wait() = 0;

  virtual ~Recver() {}
};

class Copier {
public:
  /*! prepare to copy n bytes
   */
  virtual void resize(const size_t n) = 0;

  /* copy from src to dst
   */
  virtual void copy(void *dst, const void *src) = 0;

  /*! block until copy is complete
   */
  virtual void wait() = 0;

  virtual ~Copier() {}
};
