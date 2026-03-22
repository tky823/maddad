/*
    for C++ extension with torch >= 2.4
*/

#include "maddad.h"
#include <torch/extension.h>
#include <vector>
#include <limits>
#include <cmath>

namespace
{
    void unbatched_decode(
        const float *logit_ptr,
        float *log_prob_ptr,
        int64_t *best_prev_ptr,
        int64_t *binarized_peaks_ptr,
        int64_t num_frames,
        int64_t num_states,
        int64_t num_fpbs,
        const int64_t *offsets_ptr,
        const int64_t *fpbs_ptr,
        const float *log_trans_ptr)
    {
        float inf = std::numeric_limits<float>::infinity();

        for (int64_t state_index = 0; state_index < num_states; state_index++)
        {
            log_prob_ptr[0 * num_states + state_index] = logit_ptr[0];
        }

        // frame_index > 0
        for (int64_t frame_index = 1; frame_index < num_frames; frame_index++)
        {
            float _logit = logit_ptr[frame_index];
            float *_prev_log_prob_ptr = log_prob_ptr + (frame_index - 1) * num_states;
            float *_curr_log_prob_ptr = log_prob_ptr + frame_index * num_states;
            int64_t *_curr_best_prev_ptr = best_prev_ptr + frame_index * num_states;

            for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
            {
                int64_t offset = offsets_ptr[fpb_index];
                int64_t fpb_size = fpbs_ptr[fpb_index];

                float best_log_prob = -inf;
                int64_t best_prev_state = 0;

                // first state: frame per beat may be changed
                for (int64_t prev_fpb_index = 0; prev_fpb_index < num_fpbs; prev_fpb_index++)
                {
                    int64_t prev_offset = offsets_ptr[prev_fpb_index];
                    int64_t prev_fpb_size = fpbs_ptr[prev_fpb_index];
                    int64_t prev_state_idx = prev_offset + prev_fpb_size - 1;

                    float trans_prob = log_trans_ptr[prev_fpb_index * num_fpbs + fpb_index];
                    float prob = _prev_log_prob_ptr[prev_state_idx] + trans_prob;

                    if (prob > best_log_prob)
                    {
                        best_log_prob = prob;
                        best_prev_state = prev_state_idx;
                    }
                }

                _curr_log_prob_ptr[offset] = best_log_prob + _logit / 2;
                _curr_best_prev_ptr[offset] = best_prev_state;

                // other states: frame per beat is not changed
                for (int64_t _state_index = 1; _state_index < fpb_size; _state_index++)
                {
                    int64_t state_index = offset + _state_index;
                    _curr_log_prob_ptr[state_index] = _prev_log_prob_ptr[state_index - 1] - _logit / 2;
                    _curr_best_prev_ptr[state_index] = state_index - 1;
                }
            }
        }

        // back track
        int64_t best_state_index = 0;
        float best_final_prob = -inf;
        float *_last_log_prob_ptr = log_prob_ptr + (num_frames - 1) * num_states;

        for (int64_t state_index = 0; state_index < num_states; state_index++)
        {
            if (_last_log_prob_ptr[state_index] > best_final_prob)
            {
                best_final_prob = _last_log_prob_ptr[state_index];
                best_state_index = state_index;
            }
        }

        std::vector<int64_t> peaks;
        for (int64_t frame_index = num_frames - 1; frame_index >= 0; frame_index--)
        {
            bool is_offset = false;

            for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
            {
                if (best_state_index == offsets_ptr[fpb_index])
                {
                    is_offset = true;
                    break;
                }
            }

            if (is_offset)
            {
                peaks.push_back(frame_index);
                binarized_peaks_ptr[frame_index] = 1;
            }

            best_state_index = best_prev_ptr[frame_index * num_states + best_state_index];
        }
    }
} // namespace (anonymous)

namespace maddad
{
    at::Tensor decode_beat_peaks_by_viterbi(
        const at::Tensor &logit, const at::Tensor &fpbs, const at::Tensor &log_transition_prob)
    {
        TORCH_CHECK(logit.dim() == 2, "logit should be 2 dim.");
        TORCH_CHECK(fpbs.dim() == 1, "fpbs should be 1 dim.");
        TORCH_CHECK(log_transition_prob.dim() == 2, "log_transition_prob should be 2 dim.");

        int64_t batch_size = logit.size(0);
        int64_t num_frames = logit.size(1);
        int64_t num_fpbs = fpbs.size(0);

        int64_t num_threads = torch::get_num_threads();
        int64_t grain_size = std::ceil((batch_size - 1.0) / num_threads) + 1;

        torch::TensorOptions int64options = torch::TensorOptions().dtype(torch::kInt64).device(logit.device());
        torch::TensorOptions float32options = torch::TensorOptions().dtype(torch::kFloat32).device(logit.device());

        int64_t *fpbs_ptr = fpbs.data_ptr<int64_t>();

        at::Tensor offsets = torch::zeros({num_fpbs}, int64options);
        int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
        int64_t num_states = 0;
        float inf = std::numeric_limits<float>::infinity();

        for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
        {
            offsets_ptr[fpb_index] = num_states;
            num_states += fpbs_ptr[fpb_index];
        }

        // transition
        float *log_transition_prob_ptr = log_transition_prob.data_ptr<float>();

        at::Tensor log_prob = torch::full({batch_size, num_frames, num_states}, -inf, float32options);
        at::Tensor best_prev_states = torch::zeros({batch_size, num_frames, num_states}, int64options);
        at::Tensor binarized_peaks = torch::full({batch_size, num_frames}, 0, int64options);

        float *logit_ptr = logit.data_ptr<float>();
        float *log_prob_ptr = log_prob.data_ptr<float>();
        int64_t *best_prev_ptr = best_prev_states.data_ptr<int64_t>();
        int64_t *binarized_peaks_ptr = binarized_peaks.data_ptr<int64_t>();

        torch::parallel_for(
            0, batch_size, grain_size,
            [&](int64_t start, int64_t end)
            {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
                {
                    float *_logit_ptr = logit_ptr + batch_idx * num_frames;
                    float *_log_prob_ptr = log_prob_ptr + batch_idx * num_frames * num_states;
                    int64_t *_best_prev_ptr = best_prev_ptr + batch_idx * num_frames * num_states;
                    int64_t *_binarized_peaks_ptr = binarized_peaks_ptr + batch_idx * num_frames;

                    unbatched_decode(
                        _logit_ptr, _log_prob_ptr, _best_prev_ptr, _binarized_peaks_ptr, num_frames, num_states, num_fpbs, offsets_ptr, fpbs_ptr, log_transition_prob_ptr);
                }
            });

        return binarized_peaks;
    }
} // namespace maddad

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(maddad, m)
{
    m.def("decode_beat_peaks_by_viterbi(Tensor logit, Tensor fpbs, Tensor log_transition_prob) -> Tensor");
}

TORCH_LIBRARY_IMPL(maddad, CPU, m)
{
    m.impl("decode_beat_peaks_by_viterbi", &maddad::decode_beat_peaks_by_viterbi);
}
