//
// Created by whl on 2025/6/12.
//

#ifndef GEX_H
#define GEX_H

#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32)
#    if defined(GEX_BUILD_SHARED)
#        define GEX_API __declspec(dllexport)
#    else
#        define GEX_API __declspec(dllimport)
#    endif
#else
#    define GEX_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef int (*gex_stream_callback)(const char * token, void * user_data);
typedef void * gex_context;

typedef GEX_API struct {
    const char * msg;
} gex_error;

GEX_API void      gex_error_set(const char * msg);
GEX_API gex_error get_last_error();

GEX_API gex_context gex_init_default(const char * model_path, const char * mmproj_path);
GEX_API gex_context gex_init_default_cpu(const char * model_path, const char * mmproj_path);
GEX_API gex_context gex_init_with_param(const char * model_path, const char * mmproj_path, int n_ctx, int use_gpu,int n_thread);
GEX_API void        gex_free(gex_context ctx);

GEX_API const char * gex_inference_path(gex_context ctx, const char * image_path);
GEX_API const char * gex_inference_path_stream(gex_context ctx, const char * image_path,gex_stream_callback cb);
GEX_API const char * gex_inference_mem(gex_context ctx, const unsigned char * buf, size_t buf_size);
GEX_API const char * gex_inference_mem_stream(gex_context ctx, const unsigned char * buf, size_t buf_size,gex_stream_callback cb);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  //GEX_H
