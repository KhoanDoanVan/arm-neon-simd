#ifndef NEON_TYPES_H
#define NEON_TYPES_H

#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>


/**
 * NEON Vector Types - ARM NEON hỗ trợ 2 loại register:
 * 
 * 64-bit (D registers): D0-D31
 *   - float32x2_t  : 2 float32  (2x32 = 64 bit)
 *   - int32x2_t    : 2 int32
 *   - int16x4_t    : 4 int16
 *   - uint8x8_t    : 8 uint8
 * 
 * 128-bit (Q registers): Q0-Q15  ← QUAN TRỌNG nhất!
 *   - float32x4_t  : 4 float32  (4x32 = 128 bit) ← Dùng nhiều nhất
 *   - int32x4_t    : 4 int32
 *   - int16x8_t    : 8 int16
 *   - uint8x16_t   : 16 uint8
*/

typedef float32x4_t neon_f32x4; // 4 floats = 128 bits
typedef float32x2_t neon_f32x2; // 2 floats = 64 bits
typedef int32x4_t neon_i32x4;
typedef uint8x16_t neon_i8x16;



#define NEON_ALIGNMENT 16 // NEON yêu cầu data align 16-byte
#define NEON_F32_LANES 4 // Số float32 trong 1 Q register


#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define NEON_AVAILABLE 1
#else
    #define NEON_AVAILABLE 0
    #error "NEON is not supported on this platform"
#endif



/**
 * Macro để align memory
 * Usage: float* data ALIGN_NEON = (float*)aligned_malloc(size);
*/
#define ALIGN_NEON __attribute__((aligned(NEON_ALIGNMENT)))

/**
 * Inline hint cho compiler - quan trọng cho performance
*/
#define NEON_INLINE inline __attribute__((always_inline))

/**
 * Prefetch data vào cache trước khi xử lý
 * Giảm cache miss, tăng tốc độ
*/
#define NEON_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)

/**
 * Likely/Unlikely hints cho branch prediction
*/
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)


/**
 * Tensor shape cho image/feature maps
 * NCHW format: [batch, channels, height, width]
 * NHWC format: [batch, height, width, channels]
*/
typedef struct
{
    int32_t n; // batch size
    int32_t c; // channels
    int32_t h; // height
    int32_t w; // width
} TensorShape;


/**
 * Aligned buffer cho NEON operations
 * Auto-manage memory với alignment
*/
typedef struct
{
    float* data ALIGN_NEON;
    size_t size;
    size_t capacity;
} AlignedBuffer;


/**
 * Convolution parameters
*/
typedef struct
{
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t padding_h;
    int32_t padding_w;
    int32_t dilation_h;
    int32_t dilation_w;
} ConvParams;


/**
 * Pooling parameters
*/
typedef struct
{
    int32_t pool_h;
    int32_t pool_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t padding_h;
    int32_t padding_w;
} PoolParams;


// PERFORMANCE METRICS
typedef struct
{
    double elapsed_ms;
    double gflops; // giga floating point operations per second
    size_t memory_bytes; // memory used
    double speedup; // measure with scaler version
} PerfMetrics;



// ERROR CODES
typedef enum {
    NEON_SUCCESS = 0,
    NEON_ERROR_NULL_POINTER = -1,
    NEON_ERROR_INVALID_SIZE = -2,
    NEON_ERROR_MISALIGNED = -3,
    NEON_ERROR_OUT_OF_MEMORY = -4,
    NEON_ERROR_INVALID_PARAM = -5
} NeonError;



/**
 * Pre-defined NEON vectors cho optimization
*/
static const float32x4_t NEON_ZEROS = {0.0f, 0.0f, 0.0f, 0.0f};
static const float32x4_t NEON_ONES = {1.0f, 1.0f, 1.0f, 1.0f};


/**
 * Calculate output size sau convolution/pooling
 * Formula: out = (in + 2*pad - kernel) / stride + 1
*/
#define CONV_OUT_SIZE(in_size, kernel, stride, padding) (((in_size) + 2*(padding) - (kernel)) / (stride) + 1)


/**
 * Check if pointer is NEON-aligned
*/
#define IS_ALIGNED(ptr) (((uintptr_t)(ptr) & (NEON_ALIGNMENT - 1)) == 0)


/**
 * Round up to nearest multiple of NEON_ALIGNMENT
*/
#define ALIGN_UP(x) (((x) + NEON_ALIGNMENT - 1) & ~(NEON_ALIGNMENT - 1))

/**
 * Min/Max macros
*/
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * Clamp value between min and max
*/
#define CLAMP(x, min, max) (MIN(MAX((x), (min)), (max)))

#endif // NEON_TYPES_H