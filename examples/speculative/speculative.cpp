#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "build-info.h"

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    params.perplexity = true; // HACK: enable logits_all = true
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);

    // load the draft model
    params.model = params.model_draft;
    std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(params);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
    }

    fflush(stderr);

    // eval the prompt with both models
    llama_eval(ctx_tgt,  inp.data(), int(inp.size() - 1), 0, params.n_threads);
    llama_eval(ctx_tgt, &inp.back(),      1, inp.size() - 1, params.n_threads);
    llama_eval(ctx_dft,  inp.data(),     int(inp.size()), 0, params.n_threads);

    // the 2 models should have the same vocab
    const int n_ctx   = llama_n_ctx(ctx_tgt);
    const int n_vocab = llama_n_vocab(ctx_tgt);
    //GGML_ASSERT(n_vocab == llama_n_vocab(ctx_dft));

    // how many tokens to draft each time
    const int n_draft = 16;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    std::vector<llama_token> drafted;

    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    for (auto & id : inp) {
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
    }

    bool has_eos = false;

    const auto t_gen_start = ggml_time_us();

    while (true) {
        n_past_dft -= drafted.size();
        LOG("drafted: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_dft, drafted));

        // sample from the drafted tokens if any
        int i_dft = 0;
        while (true) {
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? n_vocab : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            float * logits = llama_get_logits(ctx_tgt) + i_dft*n_vocab;
            llama_token id = 0;

            {
                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl(ctx_tgt)];
                auto last_n_repeat = std::min(std::min((int) last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx_tgt, &cur_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx_tgt, &cur_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    for (size_t idx = 0; idx < cur_p.size; idx++) {
                        if (cur_p.data[idx].id == llama_token_nl(ctx_tgt)) {
                            cur_p.data[idx].logit = nl_logit;
                            break;
                        }
                    }
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx_tgt, &cur_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx_tgt, &cur_p, temp);
                        id = llama_sample_token_mirostat(ctx_tgt, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx_tgt, &cur_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx_tgt, &cur_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k      (ctx_tgt, &cur_p, top_k, 1);
                        llama_sample_tail_free  (ctx_tgt, &cur_p, tfs_z, 1);
                        llama_sample_typical    (ctx_tgt, &cur_p, typical_p, 1);
                        llama_sample_top_p      (ctx_tgt, &cur_p, top_p, 1);
                        llama_sample_temperature(ctx_tgt, &cur_p, temp);

                        {
                            const int n_top = 10;
                            LOG("top %d candidates:\n", n_top);

                            for (int i = 0; i < n_top; i++) {
                                const llama_token id = cur_p.data[i].id;
                                LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx_tgt, id).c_str(), cur_p.data[i].p);
                            }
                        }

                        id = llama_sample_token(ctx_tgt, &cur_p);

                        LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_tgt, id).c_str());
                    }
                }

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, last_n_tokens));
            }

            const std::string token_str = llama_token_to_piece(ctx_tgt, id);
            printf("%s", token_str.c_str());
            fflush(stdout);

            if (id == llama_token_eos(ctx_tgt)) {
                has_eos = true;
            }

            ++n_predict;

            if (i_dft < (int) drafted.size() && id == drafted[i_dft]) {
                LOG("drafted token %d accepted\n", id);
                ++n_accept;
                ++n_past_tgt;
                ++n_past_dft;
                ++i_dft;

                continue;
            }

            // the drafted token was rejected or we are out of drafted tokens
            llama_eval(ctx_dft, &id, 1, n_past_dft, params.n_threads);
            ++n_past_dft;

            drafted.clear();
            drafted.push_back(id);

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        // sample n_draft tokens from the draft model picking the best token
        for (int i = 0; i < n_draft; ++i) {
            float * logits = llama_get_logits(ctx_dft);

            int   best_id     = -1;
            float best_logit  = -1e30f;
            float best_logit2 = -1e30f;
            for (int j = 0; j < n_vocab; ++j) {
                if (logits[j] > best_logit) {
                    best_logit2 = best_logit;
                    best_logit  = logits[j];
                    best_id     = j;
                }
            }

            // very low confidence in the best token
            // TODO: better way to do this
            if (best_logit - best_logit2 < 1.0f) {
                break;
            }

            drafted.push_back(best_id);
            ++n_drafted;

            llama_eval(ctx_dft, &drafted.back(), 1, n_past_dft, params.n_threads);
            ++n_past_dft;
        }

        // evaluate the target model on the drafted tokens
        llama_eval(ctx_tgt, drafted.data(), drafted.size(), n_past_tgt, params.n_threads);
        ++n_past_tgt;

        drafted.erase(drafted.begin());
    }

    auto t_gen_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("generated %d tokens in %.3f seconds, speed: %.3f t/s\n", n_predict, (t_gen_end - t_gen_start) / 1e6f, n_predict / ((t_gen_end - t_gen_start) / 1e6f));

    // TODO: make sure these numbers are computed correctly
    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ndraft:\n");
    llama_print_timings(ctx_dft);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
