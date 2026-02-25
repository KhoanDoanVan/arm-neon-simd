#ifndef MEMORY_ALIGN_H
#define MEMORY_ALIGN_H

#include "neon_types.h"
#include <stdlib.h>
#include <string.h>


/**
 * TẠI SAO CẦN ALIGNMENT?
 * 
 * 1. NEON load/store instructions (vld1q, vst1q) YÊU CẦU data phải align 16-byte
 * 2. Unaligned access:
 *    - Chậm hơn 2-3x
 *    - Có thể gây crash trên một số CPU
 * 3. Cache efficiency: Aligned data fit tốt hơn trong cache lines (64 bytes)
 * 
 * VÍ DỤ:
 * 
 * Bad (unaligned):
 *   float* data = malloc(100 * sizeof(float));  // Có thể align bất kỳ
 *   vld1q_f32(data);  // ← CÓ THỂ CHẬM hoặc CRASH!
 * 
 * Good (aligned):
 *   float* data = aligned_alloc(16, 100 * sizeof(float));
 *   vld1q_f32(data);  // ← NHANH!
*/

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate aligned memory
 * 
 * @param size: Số bytes cần allocate
 * @param alignment: Alignment boundary (thường là 16 cho NEON)
 * @return: Pointer đến aligned memory, hoặc NULL nếu fail
 * 
 * Example:
 *   float* data = (float*)aligned_malloc(1024 * sizeof(float), 16);
 *   // ... use data ...
 *   aligned_free(data);
*/
static inline void* aligned_malloc(size_t size, size_t alignment) {

    void* ptr = NULL;

    // posix_memalign yêu cầu alignment phải là power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    #if defined(_WIN32) || defined(_WIN64)
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return NULL;
        }
    #endif

    return ptr;
}


/**
 * Free aligned memory
*/
static inline void aligned_free(void* ptr) {
    if (ptr == NULL) return;

    #if defined(_WIN32) || defined(_WIN64)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}


/**
 * Allocate NEON-aligned memory (16-byte aligned)
 * Wrapper function cho convenience
*/
static inline void* neon_malloc(size_t size) {
    return aligned_malloc(size, NEON_ALIGNMENT);
}


/**
 * Free NEON-aligned memory
*/
static inline void neon_free(void* ptr) {
    aligned_free(ptr);
}


// ALIGNED BUFFER OPERATIONS
/**
 * Create aligned buffer
 * 
 * @param capacity: Số elements tối đa
 * @return: AlignedBuffer structure
 * 
 * Example:
 *   AlignedBuffer buf = aligned_buffer_create(1024);
 *   // ... use buf.data ...
 *   aligned_buffer_destroy(&buf);
*/
static inline AlignedBuffer aligned_buffer_create(size_t capacity) {
    AlignedBuffer buffer;
    buffer.capacity = capacity;
    buffer.size = 0;
    buffer.data = (float*)neon_malloc(capacity * sizeof(float));

    if (buffer.data == NULL) {
        buffer.capacity = 0;
    }

    return buffer;
}


/**
 * Destroy aligned buffer
*/
static inline void aligned_buffer_destroy(AlignedBuffer* buffer) {
    if (buffer == NULL) return;

    if (buffer->data != NULL) {
        neon_free(buffer->data);
        buffer->data == NULL;
    }

    buffer->size = 0;
    buffer->capacity = 0;
}

/**
 * Resize aligned buffer
 * Giữ nguyên data nếu có thể
*/
static inline int aligned_buffer_resize(AlignedBuffer* buffer, size_t new_capacity) {
    if (buffer == NULL) return NEON_ERROR_NULL_POINTER;

    if (new_capacity <= buffer->capacity) {
        return NEON_SUCCESS; // Không cần resize
    }

    float* new_data = (float*)neon_malloc(new_capacity * sizeof(float));
    
    if (new_data == NULL) {
        return NEON_ERROR_OUT_OF_MEMORY;
    }

    // Copy old data
    if (buffer->data != NULL) {
        memcpy(new_data, buffer->data, buffer->size * sizeof(float));
        neon_free(buffer->data);
    }

    buffer->data = new_data;
    buffer->capacity = new_capacity;

    return NEON_SUCCESS;
}


/**
 * Clear buffer (set to zero)
*/
static inline void aligned_buffer_clear(AlignedBuffer* buffer) {
    if (buffer == NULL || buffer->data == NULL) return;
    memset(buffer->data, 0, buffer->capacity * sizeof(float));
    buffer->size = 0;
}


// MEMORY COPY OPTIMIZED WITH NEON
/**
 * Fast memory copy using NEON
 * ~2x nhanh hơn memcpy() cho large arrays
 * 
 * @param dst: Destination (phải NEON-aligned)
 * @param src: Source (phải NEON-aligned)
 * @param size: Số elements (float)
*/
static inline void neon_memory_f32(float* ALIGN_NEON dst, const float* ALIGN_NEON src, size_t size) {
    if (!IS_ALIGNED(dst) || !IS_ALIGNED(src)) {
        // Fallback to standard memcpy nếu không aligned
        memcpy(dst, src, size * sizeof(float));
        return;
    }

    size_t i = 0;

    // Process 16 floats per iteration (4 NEON registers)
    // Unroll loop để tận dụng instruction pipelining
    for (; i + 16 <= size; i += 16) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float32x4_t v2 = vld1q_f32(src + i + 8);
        float32x4_t v3 = vld1q_f32(src + i + 12);

        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
        vst1q_f32(dst + i + 8, v2);
        vst1q_f32(dst + i + 12, v3);
    }

    // Process 4 floats per iteration
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1q_f32(dst + i, v);
    }

    // Handle remaining elements
    for (; i < size; i++) {
        dst[i] = src[i];
    }
}


/**
 * Fill array với một giá trị sử dụng NEON
 * ~4x nhanh hơn loop thường
*/
static inline void neon_fill_f32(float* ALIGN_NEON dst, float value, size_t size) {
    if (!IS_ALIGNED(dst)) {
        // Fallback
        for (size_t i = 0; i < size; i++) {
            dst[i] = value;
        }
        return;
    }

    // Broadcast value to all 4 lanes
    float32x4_t v = vdupq_n_f32(value);

    size_t i = 0;

    // Process 16 floats per iteration
    for (; i + 16 <= size; i += 16) {
        vst1q_f32(dst + i, v);
        vst1q_f32(dst + i + 4, v);
        vst1q_f32(dst + i + 8, v);
        vst1q_f32(dst + i + 12, v);
    }

    // Process 4 floats per iteration
    for (; i + 4 <= size; i += 4) {
        vst1q_f32(dst + i, v);
    }

    // Handle remaining elements
    for (; i < size; i++) {
        dst[i] = value;
    }
}




#ifdef __cplusplus
}
#endif

#endif