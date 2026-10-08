#include "common.h"
#include <cassert>
#include <iostream>

int main() {
  MetaInfoToExchange advertised;
  advertised.flag = ChannelType::Control;
  advertised.zip_protocol = kZipProtocol;
  advertised.ack_ring_meta.addr = 64;
  advertised.ack_ring_meta.length = 64 * 128;
  auto bytes = serialize(advertised);
  auto peer = deserialize<MetaInfoToExchange>(bytes);
  assert(peer.zip_protocol == kZipProtocol);
  assert(peer.ack_ring_meta.addr == 64 &&
         peer.ack_ring_meta.length == 64 * 128);

  // Real legacy wire extent, including the entire final RemoteMemInfo and
  // its tail padding. No legacy byte can populate the appended version.
  bytes.resize(offsetof(MetaInfoToExchange, zip_protocol));
  auto legacy = deserialize<MetaInfoToExchange>(bytes);
  assert(legacy.zip_protocol == 0);
  assert(legacy.ack_ring_meta.addr == advertised.ack_ring_meta.addr);
  assert(legacy.ack_ring_meta.length == advertised.ack_ring_meta.length);
  assert(legacy.zip_protocol != kZipProtocol);

  WriteReqMeta metadata{};
  metadata.wr_id = 0;
  metadata.ack_slot = 127;
  metadata.ack_token = 0x123456789abcdef;
  auto received = deserialize<WriteReqMeta>(serialize(metadata));
  assert(sizeof(WriteReqMeta) == 64 &&
         received.ack_token == metadata.ack_token);
  assert(received.wr_id == 0 && received.ack_slot == 127);
  std::cout << "{\"v2_roundtrip\":\"PASS\",\"legacy_version\":0,"
               "\"metadata_bytes\":64,\"token_roundtrip\":\"PASS\"}\n";
}
