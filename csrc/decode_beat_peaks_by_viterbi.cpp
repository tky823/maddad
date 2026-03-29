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
        const float *beat_log_prob_ptr,
        const float *nonbeat_log_prob_ptr,
        int8_t *peaks_ptr,
        int64_t num_frames,
        int64_t num_states,
        int64_t num_fpbs,
        const int64_t *offsets_ptr,
        const int64_t *fpbs_ptr,
        const float *log_transition_prob_ptr)
    {
        float inf = std::numeric_limits<float>::infinity();

        std::vector<int64_t> best_prev(num_frames * num_states, 0);
        std::vector<float> score(num_frames * num_states, -inf);
        int64_t *best_prev_ptr = best_prev.data();
        float *score_ptr = score.data();

        // frame_index = 0: initial distribution
        for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
        {
            int64_t offset = offsets_ptr[fpb_index];
            int64_t fpb = fpbs_ptr[fpb_index];

            // first state
            score_ptr[offset] = beat_log_prob_ptr[0];

            // other states
            for (int64_t _state_index = 1; _state_index < fpb; _state_index++)
            {
                int64_t state_index = offset + _state_index;
                score_ptr[state_index] = nonbeat_log_prob_ptr[0];
            }
        }

        // frame_index > 0
        for (int64_t frame_index = 1; frame_index < num_frames; frame_index++)
        {
            float _beat_log_prob = beat_log_prob_ptr[frame_index];
            float _nonbeat_log_prob = nonbeat_log_prob_ptr[frame_index];
            float *_prev_score_ptr = score_ptr + (frame_index - 1) * num_states;
            float *_score_ptr = score_ptr + frame_index * num_states;
            int64_t *_best_prev_ptr = best_prev_ptr + frame_index * num_states;

            for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
            {
                int64_t offset = offsets_ptr[fpb_index];
                int64_t fpb = fpbs_ptr[fpb_index];

                float best_score = -inf;
                int64_t best_prev_state = 0;

                // first state: frame per beat may be changed
                for (int64_t prev_fpb_index = 0; prev_fpb_index < num_fpbs; prev_fpb_index++)
                {
                    int64_t prev_offset = offsets_ptr[prev_fpb_index];
                    int64_t prev_fpb = fpbs_ptr[prev_fpb_index];
                    int64_t prev_state_idx = prev_offset + prev_fpb - 1;

                    float _log_transition_prob = log_transition_prob_ptr[prev_fpb_index * num_fpbs + fpb_index];
                    float _score = _prev_score_ptr[prev_state_idx] + _log_transition_prob;

                    if (_score > best_score)
                    {
                        best_score = _score;
                        best_prev_state = prev_state_idx;
                    }
                }

                _score_ptr[offset] = best_score + _beat_log_prob;
                _best_prev_ptr[offset] = best_prev_state;

                // other states: frame per beat is not changed
                for (int64_t state_index = 1; state_index < fpb; state_index++)
                {
                    int64_t _state_index = offset + state_index;
                    _score_ptr[_state_index] = _prev_score_ptr[_state_index - 1] + _nonbeat_log_prob;
                    _best_prev_ptr[_state_index] = _state_index - 1;
                }
            }
        }

        // back track
        int64_t best_state_index = 0;
        float best_final_prob = -inf;
        float *_last_score_ptr = score_ptr + (num_frames - 1) * num_states;

        for (int64_t state_index = 0; state_index < num_states; state_index++)
        {
            if (_last_score_ptr[state_index] > best_final_prob)
            {
                best_final_prob = _last_score_ptr[state_index];
                best_state_index = state_index;
            }
        }

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
                peaks_ptr[frame_index] = 1;
            }

            best_state_index = best_prev_ptr[frame_index * num_states + best_state_index];
        }
    }
} // namespace (anonymous)

namespace maddad
{
    at::Tensor decode_beat_peaks_by_viterbi(
        const at::Tensor &beat_log_prob, const at::Tensor &nonbeat_log_prob, const at::Tensor &lengths, const at::Tensor &fpbs, const at::Tensor &log_transition_prob)
    {
        TORCH_CHECK(beat_log_prob.dim() == 2, "beat_log_prob should be 2 dim.");
        TORCH_CHECK(nonbeat_log_prob.dim() == 2, "nonbeat_log_prob should be 2 dim.");
        TORCH_CHECK(nonbeat_log_prob.size(0) == beat_log_prob.size(0), "Shape of nonbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(nonbeat_log_prob.size(1) == beat_log_prob.size(1), "Shape of nonbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(lengths.dim() == 1, "lengths should be 1 dim.");
        TORCH_CHECK(lengths.size(0) == beat_log_prob.size(0), "Batch size of lengths should be same as beat_log_prob.");
        TORCH_CHECK(fpbs.dim() == 1, "fpbs should be 1 dim.");
        TORCH_CHECK(log_transition_prob.dim() == 2, "log_transition_prob should be 2 dim.");

        TORCH_CHECK(beat_log_prob.is_contiguous(), "beat_log_prob must be contiguous.");
        TORCH_CHECK(nonbeat_log_prob.is_contiguous(), "nonbeat_log_prob must be contiguous.");
        TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous.");
        TORCH_CHECK(fpbs.is_contiguous(), "fpbs must be contiguous.");
        TORCH_CHECK(log_transition_prob.is_contiguous(), "log_transition_prob must be contiguous.");

        int64_t batch_size = beat_log_prob.size(0);
        int64_t num_frames = beat_log_prob.size(1);
        int64_t num_fpbs = fpbs.size(0);

        int64_t num_threads = torch::get_num_threads();
        int64_t grain_size = std::ceil((batch_size - 1.0) / num_threads) + 1;

        torch::TensorOptions int8options = torch::TensorOptions().dtype(torch::kInt8).device(beat_log_prob.device());
        torch::TensorOptions int64options = torch::TensorOptions().dtype(torch::kInt64).device(beat_log_prob.device());
        torch::TensorOptions float32options = torch::TensorOptions().dtype(torch::kFloat32).device(beat_log_prob.device());

        int64_t *fpbs_ptr = fpbs.data_ptr<int64_t>();

        at::Tensor offsets = torch::zeros({num_fpbs}, int64options);
        int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
        int64_t num_states = 0;

        for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
        {
            offsets_ptr[fpb_index] = num_states;
            num_states += fpbs_ptr[fpb_index];
        }

        // transition
        float *log_transition_prob_ptr = log_transition_prob.data_ptr<float>();

        at::Tensor peaks = torch::full({batch_size, num_frames}, 0, int8options);

        float *beat_log_prob_ptr = beat_log_prob.data_ptr<float>();
        float *nonbeat_log_prob_ptr = nonbeat_log_prob.data_ptr<float>();
        int64_t *lengths_ptr = lengths.data_ptr<int64_t>();
        int8_t *peaks_ptr = peaks.data_ptr<int8_t>();

        torch::parallel_for(
            0, batch_size, grain_size,
            [&](int64_t start, int64_t end)
            {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
                {
                    float *_beat_log_prob_ptr = beat_log_prob_ptr + batch_idx * num_frames;
                    float *_nonbeat_log_prob_ptr = nonbeat_log_prob_ptr + batch_idx * num_frames;
                    int64_t _num_frames = lengths_ptr[batch_idx]; // actual number of frames for sample
                    int8_t *_peaks_ptr = peaks_ptr + batch_idx * num_frames;

                    unbatched_decode(
                        _beat_log_prob_ptr,
                        _nonbeat_log_prob_ptr,
                        _peaks_ptr,
                        _num_frames,
                        num_states,
                        num_fpbs,
                        offsets_ptr,
                        fpbs_ptr,
                        log_transition_prob_ptr);
                }
            });

        return peaks;
    }
} // namespace maddad

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY_FRAGMENT(maddad, m)
{
    m.def("decode_beat_peaks_by_viterbi(Tensor beat_log_prob, Tensor nonbeat_log_prob, Tensor lengths, Tensor fpbs, Tensor log_transition_prob) -> Tensor");
}

TORCH_LIBRARY_IMPL(maddad, CPU, m)
{
    m.impl("decode_beat_peaks_by_viterbi", &maddad::decode_beat_peaks_by_viterbi);
}
