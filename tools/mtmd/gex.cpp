//
// Created by whl on 2025/6/12.
//

// --- Standard Library Includes ---
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// --- llama.cpp Core Includes ---
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "gex.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

static gex_error last_error;

void gex_error_set(const char * msg) {
    last_error.msg = msg;
}

gex_error get_last_error() {
    return last_error;
}

struct gex_ctx_internal {
    mtmd::context_ptr    ctx_vision;
    common_init_result   llama_init;

    llama_model *       model;
    llama_context *     lctx;
    const llama_vocab * vocab;
    llama_batch         batch;
    int                 n_batch;

    common_sampler * smpl;

    int       n_threads = 1;
    llama_pos n_past    = 0;
    uint32_t  n_gen     = 512;

    mtmd::bitmaps bitmaps;

    std::string result;

    bool use_gpu = false;

    bool load_media_from_file(const char * fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }

    bool load_media_from_mem(const unsigned char * data_ptr, const size_t size) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(ctx_vision.get(), data_ptr, size));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }

    const char * inference(gex_stream_callback cb = nullptr) {
        {
            constexpr auto  prompt = R"(<__media__>:)";
            mtmd_input_text text;
            text.text          = prompt;
            text.add_special   = false;
            text.parse_special = true;

            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto               bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t            res           = mtmd_tokenize(ctx_vision.get(),
                                                             chunks.ptr.get(),  // output
                                                             &text,             // text
                                                             bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
            bitmaps.entries.clear();
            if (res != 0) {
                gex_error_set("Inference tokenizer failed");
                return nullptr;
            }
            llama_pos new_n_past;
            if (mtmd_helper_eval_chunks(ctx_vision.get(),
                                        lctx,              // lctx
                                        chunks.ptr.get(),  // chunks
                                        n_past,            // n_past
                                        0,                 // seq_id
                                        n_batch,           // n_batch
                                        true,              // logits_last
                                        &new_n_past)) {
                gex_error_set("Unable to eval prompt\n");
                return nullptr;
            }
            n_past = new_n_past;
        }
        {
            llama_tokens generated_tokens;

            for (int i = 0; i < n_gen; i++) {
                llama_token token_id = common_sampler_sample(smpl, lctx, -1);
                generated_tokens.push_back(token_id);
                common_sampler_accept(smpl, token_id, true);
                if (llama_vocab_is_eog(vocab, token_id)) {
                    break;
                }
                auto s = common_token_to_piece(lctx, token_id);
                result += s;
                if (cb) {
                    if (cb(s.c_str(),nullptr) != 0) {
                        break;
                    }
                }

                common_batch_clear(batch);
                common_batch_add(batch, token_id, n_past++, { 0 }, true);
                if (llama_decode(lctx, batch)) {
                    gex_error_set("failed to decode token\n");
                    throw std::runtime_error("failed to decode token");
                    return nullptr;
                }
            }
        }

        return result.c_str();
    }

    void clear_ctx() {
        n_past = 0;
        result.clear();
        result.reserve(1024);
        llama_memory_seq_rm(llama_get_memory(lctx), 0, 0, -1);
        mtmd_reset(ctx_vision.get(), use_gpu);
    }

    ~gex_ctx_internal() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }
};

gex_context init_ctx(gex_ctx_internal * ctx, common_params params, const char * mmproj_path) {
    common_log_set_verbosity_thold(-1); // disable logs
    llama_log_set([](ggml_log_level, const char*, void*) {},nullptr);

    ctx->llama_init = common_init_from_params(params);
    common_init();

    ctx->lctx         = ctx->llama_init.context.get();
    ctx->model        = ctx->llama_init.model.get();
    ctx->n_gen        = 512;
    ctx->n_batch      = params.n_batch;
    ctx->n_past       = 0;
    if (!ctx->llama_init.model || !ctx->llama_init.context) {
        gex_error_set("Init params failed");
        delete ctx;
        return nullptr;  // Failed to load LLM
    }

    ctx->smpl = common_sampler_init(ctx->model, params.sampling);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu             = params.mmproj_use_gpu;
    mparams.print_timings       = false;
    mparams.n_threads           = params.cpuparams.n_threads;
    mparams.verbosity           = GGML_LOG_LEVEL_ERROR;

    ctx->ctx_vision.reset(mtmd_init_from_file(mmproj_path, ctx->llama_init.model.get(), mparams));
    if (!ctx->ctx_vision.get()) {
        gex_error_set("Init vision model failed");
        delete ctx;
        return nullptr;  // Failed to load vision projector
    }

    ctx->vocab     = llama_model_get_vocab(ctx->llama_init.model.get());
    ctx->batch     = llama_batch_init(1, 0, 1);
    ctx->n_threads = params.cpuparams.n_threads;
    return ctx;
}

gex_context gex_init_default(const char * model_path, const char * mmproj_path) {
    auto *         ctx     = new gex_ctx_internal;
    constexpr bool use_gpu = true;
    ctx->use_gpu           = use_gpu;

    common_params params;
    params.model.path          = model_path;
    params.mmproj.path         = mmproj_path;
    params.mmproj_use_gpu      = use_gpu;
    params.cpuparams.n_threads = std::max(static_cast<int>(std::thread::hardware_concurrency()),8);
    params.n_gpu_layers        = use_gpu ? 99 : 0;
    params.n_ctx               = 4096;
    // params.n_batch             = 32;
    params.n_batch             = 2048;
    params.n_ubatch            = 512;
    params.n_predict           = 512;
    params.sampling.temp       = -1.;
    params.flash_attn          = true;
    params.verbosity           = -1;

    return init_ctx(ctx, params, mmproj_path);
}

gex_context gex_init_default_cpu(const char * model_path, const char * mmproj_path) {
    auto *         ctx     = new gex_ctx_internal;
    constexpr bool use_gpu = false;
    ctx->use_gpu           = use_gpu;

    common_params params;
    params.model.path          = model_path;
    params.mmproj.path         = mmproj_path;
    params.mmproj_use_gpu      = use_gpu;
    params.cpuparams.n_threads = std::thread::hardware_concurrency();
    params.n_gpu_layers        = use_gpu ? 99 : 0;
    params.n_ctx               = 4096;
    // params.n_batch             = 32;
    params.n_batch             = 2048;
    params.n_ubatch            = 512;
    params.n_predict           = 512;
    params.sampling.temp       = -1.;
    params.verbosity           = -1;

    return init_ctx(ctx, params, mmproj_path);
}

gex_context gex_init_with_param(const char * model_path, const char * mmproj_path, int n_ctx, int use_gpu,
                                int n_thread) {
    auto ctx     = new gex_ctx_internal;
    ctx->use_gpu = use_gpu;

    common_params params;
    params.model.path          = model_path;
    params.mmproj.path         = mmproj_path;
    params.mmproj_use_gpu      = use_gpu;
    params.cpuparams.n_threads = n_thread;
    params.n_gpu_layers        = use_gpu ? 99 : 0;
    params.n_ctx               = n_ctx + 256;
    // params.n_batch             = 32;
    params.n_predict           = n_ctx;
    params.sampling.temp       = 0.;
    params.verbosity           = -1;


    return init_ctx(ctx, params, mmproj_path);
}

void gex_free(gex_context ctx) {
    delete static_cast<gex_ctx_internal *>(ctx);
}

const char * gex_inference_path(gex_context ctx, const char * image_path) {
    auto c = static_cast<gex_ctx_internal *>(ctx);
    c->clear_ctx();

    if (!c->load_media_from_file(image_path)) {
        gex_error_set("load image from path failed");
        return nullptr;
    }
    const auto * res = c->inference();
    return res;
}

const char * gex_inference_mem(gex_context ctx, const unsigned char * buf, size_t buf_size) {
    auto c = static_cast<gex_ctx_internal *>(ctx);
    c->clear_ctx();

    c->load_media_from_mem(buf, buf_size);
    const auto * res = c->inference();
    return res;
}


const char * gex_inference_path_stream(gex_context ctx, const char * image_path, gex_stream_callback cb) {
    auto c = static_cast<gex_ctx_internal *>(ctx);
    c->clear_ctx();

    if (!c->load_media_from_file(image_path)) {
        gex_error_set("load image from path failed");
        return nullptr;
    }
    const auto * res = c->inference(cb);
    return res;
}

const char * gex_inference_mem_stream(gex_context ctx, const unsigned char * buf, size_t buf_size,
                                      gex_stream_callback cb) {
    auto c = static_cast<gex_ctx_internal *>(ctx);
    c->clear_ctx();

    c->load_media_from_mem(buf, buf_size);
    const auto * res = c->inference(cb);
    return res;
}
