#include "compression.h"
#include "dietgpu_decompress.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>

size_t get_min_compress_bytes_from_env() {
  char const* env = std::getenv("UCCL_P2P_MIN_COMPRESS_BYTES");
  if (!env) return 16ull * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 16ull * 1024 * 1024;
  return static_cast<size_t>(v);
}

size_t const& kMinCompressBytes = []() -> size_t const& {
  static size_t v = get_min_compress_bytes_from_env();
  return v;
}();

size_t get_compress_buffer_bytes_from_env() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_BUFFER_BYTES");
  if (!env) return 2ull * 1024 * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 2ull * 1024 * 1024 * 1024;
  return static_cast<size_t>(v);
}

size_t const& kCompressBufferSize = []() -> size_t const& {
  static size_t v = get_compress_buffer_bytes_from_env();
  return v;
}();

size_t const kDecompressBufferSize = [] {
  char const* env = std::getenv("UCCL_P2P_DECOMPRESS_BUFFER_BYTES");
  if (!env) return kCompressBufferSize;
  return static_cast<size_t>(std::stoull(env));
}();

size_t const kZipMaxPeers = [] {
  char const* env = std::getenv("UCCL_P2P_ZIP_MAX_PEERS");
  size_t peers = env ? std::stoull(env) : 4;
  if (!peers || peers > kAckRingDepth / 64)
    throw std::invalid_argument("Zip max peers must be between 1 and 64");
  return peers;
}();

CompressStrategy get_compress_strategy_from_env() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_STRATEGY");

  if (!env || env[0] == '\0') {
    return CompressStrategy::kNone;
  }

  std::string s(env);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (s == "none" || s == "off" || s == "0") {
    return CompressStrategy::kNone;
  }

  if (s == "split" || s == "split_only") {
    return CompressStrategy::kSplitOnly;
  }

  if (s == "encode" || s == "split_encode" || s == "full" || s == "1") {
    return CompressStrategy::kSplitEncode;
  }

  return CompressStrategy::kSplitEncode;
}

size_t max_compressed_size(FloatType float_type, size_t bytes) {
#ifdef USE_DIETGPU
  auto type = to_dietgpu(float_type);
  if (type == dietgpu::FloatType::kUndefined ||
      bytes > std::numeric_limits<uint32_t>::max())
    return 0;
  size_t word_bytes = type == dietgpu::FloatType::kFloat32 ? 4
                      : dietgpu::isFloat8Type(type)        ? 1
                                                           : 2;
  if (bytes == 0 || bytes % word_bytes != 0 ||
      (dietgpu::isFloat8Type(type) && bytes % 2 != 0))
    return 0;
  size_t bound = dietgpu::getMaxFloatCompressedSize(
      type, static_cast<uint32_t>(bytes / word_bytes));
  // The bound includes headers and worst-case ANS expansion; an overflowed
  // 32-bit codec size must not become a smaller arena reservation.
  return bound >= bytes ? bound : 0;
#else
  (void)float_type;
  (void)bytes;
  return 0;
#endif
}

#if defined USE_DIETGPU
dietgpu::FloatType to_dietgpu(FloatType t) {
  switch (t) {
    case FloatType::kFloat16:
      return dietgpu::FloatType::kFloat16;
    case FloatType::kBFloat16:
      return dietgpu::FloatType::kBFloat16;
    case FloatType::kFloat32:
      return dietgpu::FloatType::kFloat32;
    case FloatType::kFloat8E4M3FN:
      return dietgpu::FloatType::kFloat8E4M3FN;
    case FloatType::kFloat8E5M2:
      return dietgpu::FloatType::kFloat8E5M2;
    case FloatType::kUndefined:
    default:
      return dietgpu::FloatType::kUndefined;
  }
}

FloatType from_dietgpu(dietgpu::FloatType t) {
  switch (t) {
    case dietgpu::FloatType::kFloat16:
      return FloatType::kFloat16;
    case dietgpu::FloatType::kBFloat16:
      return FloatType::kBFloat16;
    case dietgpu::FloatType::kFloat32:
      return FloatType::kFloat32;
    case dietgpu::FloatType::kFloat8E4M3FN:
      return FloatType::kFloat8E4M3FN;
    case dietgpu::FloatType::kFloat8E5M2:
      return FloatType::kFloat8E5M2;
    default:
      return FloatType::kUndefined;
  }
}

FloatCompressCtx::FloatCompressCtx() = default;

FloatCompressCtx::FloatCompressCtx(FloatType ft)
    : dietgpu::FloatCompressSplitContext(to_dietgpu(ft)) {}

FloatCompressCtx::~FloatCompressCtx() {
  if (raw_params_dev) {
    // dietgpu's ~GpuMemoryReservation does `if (ptr) CHECK(res)`. Our
    // hand-constructed params_dev has res=nullptr, so clear ptr first to
    // disarm the assertion; we own the buffer and free it directly.
    params_dev.ptr = nullptr;
    params_dev.res = nullptr;
    gpuFree(raw_params_dev);
    raw_params_dev = nullptr;
  }
}

FloatType FloatCompressCtx::get_float_type() const {
  return from_dietgpu(float_type);
}

size_t FloatCompressCtx::get_max_size() const { return maxSize; }

CompressCtx make_compress_ctx(FloatType ft) {
  return std::make_shared<FloatCompressCtx>(ft);
}

#else

DummyCompressCtx::DummyCompressCtx() = default;

DummyCompressCtx::DummyCompressCtx(FloatType ft)
    : float_type(ft), max_size(0) {}

FloatType DummyCompressCtx::get_float_type() const { return float_type; }

size_t DummyCompressCtx::get_max_size() const { return max_size; }

CompressCtx make_compress_ctx(FloatType ft) {
  return std::make_shared<DummyCompressCtx>(ft);
}

#endif

NullCompressorBackend::NullCompressorBackend()
    : compress_strategy_(CompressStrategy::kNone) {}

NullCompressorBackend::~NullCompressorBackend() = default;

std::shared_ptr<RegMemBlock> NullCompressorBackend::get_compress_buffer()
    const {
  return nullptr;
}

std::shared_ptr<RegMemBlock> NullCompressorBackend::get_decompress_buffer()
    const {
  return nullptr;
}

bool NullCompressorBackend::compress(std::shared_ptr<RDMASendRequest> /*req*/) {
  return false;
}

bool NullCompressorBackend::decompress(RemoteMemInfo const& /*input*/,
                                       RegMemBlock& /*output*/,
                                       FloatType /*float_type*/) {
  return false;
}

std::unique_ptr<DecompressRequest> NullCompressorBackend::decompress_async(
    RemoteMemInfo const& /*input*/, RegMemBlock& /*output*/,
    FloatType /*float_type*/) {
  return nullptr;
}

bool NullCompressorBackend::should_compress(size_t /*size*/) { return false; }

bool NullCompressorBackend::should_compress_and_split_first(size_t /*size*/) {
  return false;
}

void NullCompressorBackend::prepare_split_context(void* /*addr*/,
                                                  size_t /*size*/,
                                                  CompressCtx /*ctx*/) {}

bool NullCompressorBackend::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> /*req*/) {
  return false;
}

uint32_t NullCompressorBackend::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> /*req*/) {
  return 0;
}

CompressStrategy NullCompressorBackend::get_compress_strategy() {
  return compress_strategy_;
}

#ifdef USE_DIETGPU
DietGPUCompressorBackend::DietGPUCompressorBackend()
    : device_(-1),
      dev_compressed_size_(nullptr),
      stream_(nullptr),
      res_(nullptr) {
  compress_strategy_ = get_compress_strategy_from_env();
  if (compress_strategy_ == CompressStrategy::kNone) {
    return;
  }
  device_ = dietgpu::getCurrentDevice();
  // Initialize green context BEFORE creating any CUDA resources (streams,
  // memory) so that they all belong to the green context.
  CodecContextScope context(device_);

  GPU_RT_CHECK(gpuMalloc(&dev_compressed_size_, sizeof(uint32_t)));
  // Initialize compression buffer
  auto allocator = std::make_shared<MemoryAllocator>();
  buffer_ = allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

  // Initialize decompression buffer
  decompressBuffer_ =
      allocator->allocate(kDecompressBufferSize, MemoryType::GPU, nullptr);

  // Initialize GPU stream
  GPU_RT_CHECK(gpuStreamCreate(&stream_));
  decompress_pool_ = make_decompress_pool(device_, kAckRingDepth);

  // Initialize StackDeviceMemory for compress/decompress operations
  res_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                        3 * kCompressBufferSize);
  // Per-ctx split metadata is now allocated/freed directly by each
  // FloatCompressCtx via gpuMalloc/gpuFree — no stack allocator needed.
}

DietGPUCompressorBackend::~DietGPUCompressorBackend() {
  if (device_ < 0) return;
  CodecContextScope context(device_);
  if (stream_) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream_));
    GPU_RT_CHECK(gpuStreamDestroy(stream_));
    stream_ = nullptr;
  }
  if (res_) {
    delete res_;
    res_ = nullptr;
  }
  if (dev_compressed_size_) {
    GPU_RT_CHECK(gpuFree(dev_compressed_size_));
  }
  decompress_pool_.reset();
  buffer_.reset();
  decompressBuffer_.reset();
}

std::shared_ptr<RegMemBlock> DietGPUCompressorBackend::get_compress_buffer()
    const {
  return buffer_;
}

std::shared_ptr<RegMemBlock> DietGPUCompressorBackend::get_decompress_buffer()
    const {
  return decompressBuffer_;
}

bool DietGPUCompressorBackend::compress(std::shared_ptr<RDMASendRequest> req) {
  if (device_ < 0) return false;
  std::lock_guard<std::mutex> lock(codec_mu_);
  CodecContextScope context(device_);
  if (unlikely(!req || !req->local_mem || !req->compress_ctx || !stream_ ||
               !res_ || !buffer_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress - Invalid parameters";
    return false;
  }
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);
  // Setup compression config
  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  // Calculate element count from bytes
  uint32_t numFloats = dietgpu::getElementCountFromBytes(
      to_dietgpu(ctx->get_float_type()), req->local_mem->size);

  // Setup batch (single element batch)
  void const* inPtrs[1] = {req->local_mem->addr};
  uint32_t inSizes[1] = {numFloats};
  void* outPtrs[1] = {buffer_->addr};

  // Compress
  dietgpu::floatCompress(*res_, compressConfig,
                         1,  // numInBatch
                         inPtrs, inSizes, outPtrs, dev_compressed_size_,
                         stream_);
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));
  // Get compressed size
  uint32_t compressedSize = 0;
  GPU_RT_CHECK(gpuMemcpy(&compressedSize, dev_compressed_size_,
                         sizeof(compressedSize), gpuMemcpyDeviceToHost));

  UCCL_LOG(INFO, UCCL_RDMA)
      << "DietGPUCompressorBackend: Compressed " << req->local_mem->size
      << " bytes to " << compressedSize << " bytes, ratio: "
      << static_cast<float>(compressedSize) /
             static_cast<float>(req->local_mem->size)
      << "x";

  // Update request to use compressed buffer
  req->local_mem = std::make_shared<RegMemBlock>(
      buffer_->addr, compressedSize, buffer_->mr_array, buffer_->type);
  return true;
}

bool DietGPUCompressorBackend::decompress(RemoteMemInfo const& input,
                                          RegMemBlock& output,
                                          FloatType float_type) {
  auto request = decompress_async(input, output, float_type);
  return request && request->wait() == DecompressStatus::kSuccess;
}

std::unique_ptr<DecompressRequest> DietGPUCompressorBackend::decompress_async(
    RemoteMemInfo const& input, RegMemBlock& output, FloatType float_type) {
  if (device_ < 0) return nullptr;
  auto ft = to_dietgpu(float_type);
  if (!stream_ || !res_ || input.addr == 0 || input.length == 0 ||
      !output.addr || output.size == 0 || ft == dietgpu::FloatType::kUndefined)
    return nullptr;
  size_t word_bytes = ft == dietgpu::FloatType::kFloat32 ? 4
                      : dietgpu::isFloat8Type(ft)        ? 1
                                                         : 2;
  if (output.size % word_bytes != 0 ||
      (dietgpu::isFloat8Type(ft) && output.size % 2 != 0))
    return nullptr;
  uint32_t num_floats = dietgpu::getElementCountFromBytes(ft, output.size);
  return make_decompress_request(
      device_, *res_, stream_, codec_mu_, decompress_pool_,
      reinterpret_cast<void const*>(input.addr), output.addr, ft, num_floats);
}

bool DietGPUCompressorBackend::should_compress(size_t size) {
  if (compress_strategy_ == CompressStrategy::kNone) {
    return false;
  }
  if (unlikely(size > buffer_->size)) {
    UCCL_LOG(ERROR) << "DietGPUCompressorBackend::should_compress: data size ("
                    << size << " bytes) exceeds compress buffer capacity ("
                    << buffer_->size << " bytes), skipping compression";
    return false;
  }
  return size >= kMinCompressBytes;
}

bool DietGPUCompressorBackend::should_compress_and_split_first(size_t size) {
  return compress_strategy_ == CompressStrategy::kSplitOnly &&
         should_compress(size);
}

void DietGPUCompressorBackend::prepare_split_context(void* addr, size_t size,
                                                     CompressCtx ctx) {
  if (device_ < 0) return;
  std::lock_guard<std::mutex> lock(codec_mu_);
  CodecContextScope context(device_);
  if (unlikely(ctx == nullptr)) {
    std::cout << "unlikely(ctx == nullptr)" << std::endl;
    return;
  }
  if (unlikely(!should_compress(size))) {
    return;
  }
  // kUndefined means no float_type was supplied; skip compression.
  if (unlikely(ctx->get_float_type() == FloatType::kUndefined)) {
    return;
  }
  auto float_ctx = std::static_pointer_cast<FloatCompressCtx>(ctx);
  dietgpu::FloatType float_type = to_dietgpu(float_ctx->get_float_type());
  uint32_t numFloats = static_cast<uint32_t>(
      dietgpu::getElementCountFromBytes(float_type, size));
  // FP8 types pack two values into one uint16 pair, so the split kernel
  // expects the pair count (half the element count).
  if (dietgpu::isFloat8Type(float_type)) {
    assert(numFloats % 2 == 0);
    numFloats /= 2;
  }

  // params_dev layout: [in_ptr, inSize, out_ptr]. Plain cudaMalloc so
  // each ctx owns its buffer independently; raw_params_dev holds ownership.
  uintptr_t hostParams[3];
  hostParams[0] = reinterpret_cast<uintptr_t>(addr);
  hostParams[1] = static_cast<uintptr_t>(numFloats);
  hostParams[2] = reinterpret_cast<uintptr_t>(buffer_->addr);

  void* raw = nullptr;
  GPU_RT_CHECK(gpuMalloc(&raw, sizeof(hostParams)));
  GPU_RT_CHECK(gpuMemcpyAsync(raw, hostParams, sizeof(hostParams),
                              gpuMemcpyHostToDevice, stream_));
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));

  if (float_ctx->raw_params_dev) {
    // Re-prepare: clear stale params_dev before reassigning, then free old
    // buffer.
    float_ctx->params_dev.ptr = nullptr;
    float_ctx->params_dev.res = nullptr;
    gpuFree(float_ctx->raw_params_dev);
  }
  float_ctx->raw_params_dev = raw;
  float_ctx->params_dev = dietgpu::GpuMemoryReservation<uintptr_t>(
      /*res=*/nullptr, dietgpu::getCurrentDevice(), stream_, raw,
      /*num=*/3, /*sizeAllocated=*/sizeof(hostParams));
  float_ctx->maxSize = size;
}

bool DietGPUCompressorBackend::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  if (device_ < 0) return false;
  std::lock_guard<std::mutex> lock(codec_mu_);
  CodecContextScope context(device_);
  if (unlikely(!req || !req->compress_ctx || !stream_ || !res_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress_split_one_batch - "
                      "Invalid parameters";
    return false;
  }
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);

  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  dietgpu::floatCompressSplitOneBatch(*res_, compressConfig, stream_, *ctx);

  int32_t data_size =
      dietgpu::roundDown(dietgpu::getUncompDataSizeFromByteSize(
                             compressConfig.floatType, ctx->maxSize),
                         ChunkSplitStrategy::kMessageChunkSizeKB);
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));
  req->local_mem = std::make_shared<RegMemBlock>(
      buffer_->addr, data_size, buffer_->mr_array, buffer_->type);
  return true;
}

uint32_t DietGPUCompressorBackend::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  if (device_ < 0) return 0;
  std::lock_guard<std::mutex> lock(codec_mu_);
  CodecContextScope context(device_);
  if (unlikely(!req || !req->compress_ctx || !stream_ || !res_ || !buffer_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress_encode_one_batch - "
                      "Invalid parameters";
    return 0;
  }
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);

  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  dietgpu::floatCompressEncodeOneBatch(*res_, compressConfig, *ctx,
                                       dev_compressed_size_, stream_);
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));
  // Get compressed size
  uint32_t compressedSize = 0;
  GPU_RT_CHECK(gpuMemcpy(&compressedSize, dev_compressed_size_,
                         sizeof(compressedSize), gpuMemcpyDeviceToHost));

  UCCL_LOG(INFO, UCCL_RDMA)
      << "DietGPUCompressorBackend: Encode done, " << req->local_mem->size
      << " bytes -> " << compressedSize << " bytes, ratio: "
      << static_cast<float>(compressedSize) /
             static_cast<float>(req->local_mem->size)
      << "x";

  // Update request to use compressed buffer
  uint32_t uncompressed_size = req->local_mem->size;
  req->local_mem = std::make_shared<RegMemBlock>(
      static_cast<char*>(buffer_->addr) + uncompressed_size,
      compressedSize - uncompressed_size, buffer_->mr_array, buffer_->type);
  req->remote_mem->addr = req->remote_mem->addr + uncompressed_size;
  ctx->histogram_dev.release();
  ctx->toComp_dev.release();
  return compressedSize;
}

CompressStrategy DietGPUCompressorBackend::get_compress_strategy() {
  return compress_strategy_;
}
#endif  // USE_DIETGPU

Compressor& Compressor::get_instance() {
  static Compressor instance;
  return instance;
}

std::shared_ptr<RegMemBlock> Compressor::get_compress_buffer() const {
  return backend_->get_compress_buffer();
}

std::shared_ptr<RegMemBlock> Compressor::get_decompress_buffer() const {
  return backend_->get_decompress_buffer();
}

std::shared_ptr<ZipRegion> Compressor::acquire_decompress_region() {
  if (!decompress_regions_) return nullptr;
  auto lease = decompress_regions_->acquire();
  return lease ? std::make_shared<ZipRegion>(get_decompress_buffer(),
                                             std::move(lease))
               : nullptr;
}

std::unique_lock<std::mutex> Compressor::lock_write_buffer() {
  return std::unique_lock<std::mutex>(write_buffer_mu_);
}

bool Compressor::compress(std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress(req);
}

bool Compressor::decompress(RemoteMemInfo const& input, RegMemBlock& output,
                            FloatType float_type) {
  return backend_->decompress(input, output, float_type);
}

std::unique_ptr<DecompressRequest> Compressor::decompress_async(
    RemoteMemInfo const& input, RegMemBlock& output, FloatType float_type) {
  return backend_->decompress_async(input, output, float_type);
}

bool Compressor::should_compress(size_t size) {
  return backend_->should_compress(size);
}

bool Compressor::should_compress_and_split_first(size_t size) {
  return backend_->should_compress_and_split_first(size);
}

void Compressor::prepare_split_context(void* addr, size_t size,
                                       CompressCtx ctx) {
  backend_->prepare_split_context(addr, size, ctx);
}

bool Compressor::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress_split_one_batch(req);
}

uint32_t Compressor::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress_encode_one_batch(req);
}

CompressStrategy Compressor::get_compress_strategy() {
  return backend_->get_compress_strategy();
}

Compressor::Compressor() {
#ifdef USE_DIETGPU
  backend_ = std::make_unique<DietGPUCompressorBackend>();
#else
  backend_ = std::make_unique<NullCompressorBackend>();
#endif
  auto buffer = get_decompress_buffer();
  if (buffer)
    decompress_regions_ =
        std::make_unique<ZipRegionPool>(buffer->size, kZipMaxPeers, 256);
}

Compressor::~Compressor() = default;
