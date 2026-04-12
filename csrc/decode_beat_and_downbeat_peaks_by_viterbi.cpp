/*
    for C++ extension with torch >= 2.4
*/

#include "maddad.h"
#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <limits>
#include <cmath>

namespace
{
    void unbatched_decode(
        const float *beat_log_prob_ptr,
        const float *downbeat_log_prob_ptr,
        const float *nonbeat_log_prob_ptr,
        float *score_ptr,
        float *best_score_ptr,
        int64_t *best_prev_ptr,
        int8_t *peaks_ptr,
        int64_t num_frames,
        int64_t num_states,
        int64_t num_fpbs,
        int64_t meter,
        const float beat_region,
        const int64_t *offsets_ptr,
        const int64_t *fpbs_ptr,
        const float *log_transition_prob_ptr)
    {
        float inf = std::numeric_limits<float>::infinity();

        // frame_index = 0: initial distribution
        for (int8_t beat = 0; beat < meter; beat++)
        {
            for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
            {
                int64_t offset = offsets_ptr[beat * num_fpbs + fpb_index];
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
        }

        // frame_index > 0
        for (int64_t frame_index = 1; frame_index < num_frames; frame_index++)
        {
            float _beat_log_prob = beat_log_prob_ptr[frame_index];
            float _downbeat_log_prob = downbeat_log_prob_ptr[frame_index];
            float _nonbeat_log_prob = nonbeat_log_prob_ptr[frame_index];
            float *_prev_score_ptr = score_ptr + (frame_index - 1) * num_states;
            float *_score_ptr = score_ptr + frame_index * num_states;
            int64_t *_best_prev_ptr = best_prev_ptr + frame_index * num_states;

            for (int8_t beat = 0; beat < meter; beat++)
            {
                const int8_t prev_beat = (beat - 1 + meter) % meter;

                for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
                {
                    int64_t offset = offsets_ptr[beat * num_fpbs + fpb_index];
                    int64_t fpb = fpbs_ptr[fpb_index];

                    float best_log_prob = -inf;
                    int64_t best_prev_state = 0;

                    // first state: frame per beat may be changed
                    for (int64_t prev_fpb_index = 0; prev_fpb_index < num_fpbs; prev_fpb_index++)
                    {
                        int64_t prev_offset = offsets_ptr[prev_beat * num_fpbs + prev_fpb_index];
                        int64_t prev_fpb = fpbs_ptr[prev_fpb_index];
                        int64_t prev_state_idx = prev_offset + prev_fpb - 1;

                        float _log_transition_prob = log_transition_prob_ptr[prev_fpb_index * num_fpbs + fpb_index];
                        float prob = _prev_score_ptr[prev_state_idx] + _log_transition_prob;

                        if (prob > best_log_prob)
                        {
                            best_log_prob = prob;
                            best_prev_state = prev_state_idx;
                        }
                    }

                    if (beat == 0)
                    {
                        _score_ptr[offset] = best_log_prob + _downbeat_log_prob;
                    }
                    else
                    {
                        _score_ptr[offset] = best_log_prob + _beat_log_prob;
                    }

                    _best_prev_ptr[offset] = best_prev_state;

                    // other states: frame per beat is not changed
                    for (int64_t state_index = 1; state_index < fpb; state_index++)
                    {
                        int64_t _state_index = offset + state_index;
                        if (static_cast<float>(state_index) / fpb < beat_region)
                        {
                            if (beat == 0)
                            {
                                _score_ptr[_state_index] = _prev_score_ptr[_state_index - 1] + _downbeat_log_prob;
                            }
                            else
                            {
                                _score_ptr[_state_index] = _prev_score_ptr[_state_index - 1] + _beat_log_prob;
                            }
                        }
                        else
                        {
                            _score_ptr[_state_index] = _prev_score_ptr[_state_index - 1] + _nonbeat_log_prob;
                        }
                        _best_prev_ptr[_state_index] = _state_index - 1;
                    }
                }
            }
        }

        // back track
        int64_t best_state_index = 0;
        *best_score_ptr = -inf;
        float *_last_score_ptr = score_ptr + (num_frames - 1) * num_states;

        for (int64_t state_index = 0; state_index < num_states; state_index++)
        {
            if (_last_score_ptr[state_index] > *best_score_ptr)
            {
                *best_score_ptr = _last_score_ptr[state_index];
                best_state_index = state_index;
            }
        }

        for (int64_t frame_index = num_frames - 1; frame_index >= 0; frame_index--)
        {
            int8_t beat = 0;
            bool is_offset = false;

            for (beat = 0; beat < meter; beat++)
            {
                for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
                {
                    if (best_state_index == offsets_ptr[beat * num_fpbs + fpb_index])
                    {
                        is_offset = true;
                        break;
                    }
                }

                if (is_offset)
                {
                    break;
                }
            }

            if (is_offset)
            {
                peaks_ptr[frame_index] = beat + 1;
            }

            best_state_index = best_prev_ptr[frame_index * num_states + best_state_index];
        }
    }
} // namespace (anonymous)

namespace maddad
{
    std::tuple<at::Tensor, at::Tensor> decode_beat_and_downbeat_peaks_by_viterbi(
        const at::Tensor &beat_log_prob,
        const at::Tensor &downbeat_log_prob,
        const at::Tensor &nonbeat_log_prob,
        const at::Tensor &lengths,
        const at::Tensor &fpbs,
        const int8_t meter,
        const double beat_region,
        const at::Tensor &log_transition_prob)
    {
        TORCH_CHECK(beat_log_prob.dim() == 2, "beat_log_prob should be 2 dim.");
        TORCH_CHECK(downbeat_log_prob.dim() == 2, "downbeat_log_prob should be 2 dim.");
        TORCH_CHECK(downbeat_log_prob.size(0) == beat_log_prob.size(0), "Shape of downbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(downbeat_log_prob.size(1) == beat_log_prob.size(1), "Shape of downbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(nonbeat_log_prob.dim() == 2, "nonbeat_log_prob should be 2 dim.");
        TORCH_CHECK(nonbeat_log_prob.size(0) == beat_log_prob.size(0), "Shape of nonbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(nonbeat_log_prob.size(1) == beat_log_prob.size(1), "Shape of nonbeat_log_prob should be same as beat_log_prob.");
        TORCH_CHECK(fpbs.dim() == 1, "fpbs should be 1 dim.");
        TORCH_CHECK(log_transition_prob.dim() == 2, "log_transition_prob should be 2 dim.");

        TORCH_CHECK(beat_log_prob.is_contiguous(), "beat_log_prob must be contiguous.");
        TORCH_CHECK(downbeat_log_prob.is_contiguous(), "downbeat_log_prob must be contiguous.");
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

        at::Tensor offsets = torch::zeros({meter * num_fpbs}, int64options);
        int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
        int64_t num_states = 0;
        float inf = std::numeric_limits<float>::infinity();

        for (int64_t beat = 0; beat < meter; beat++)
        {
            for (int64_t fpb_index = 0; fpb_index < num_fpbs; fpb_index++)
            {
                offsets_ptr[beat * num_fpbs + fpb_index] = num_states;
                num_states += fpbs_ptr[fpb_index];
            }
        }

        // transition
        float *log_transition_prob_ptr = log_transition_prob.data_ptr<float>();

        at::Tensor score = torch::full({batch_size, num_frames, num_states}, -inf, float32options);
        at::Tensor best_score = torch::full({batch_size}, -inf, float32options);
        at::Tensor best_prev_states = torch::zeros({batch_size, num_frames, num_states}, int64options);
        at::Tensor peaks = torch::full({batch_size, num_frames}, 0, int8options);

        float *beat_log_prob_ptr = beat_log_prob.data_ptr<float>();
        float *downbeat_log_prob_ptr = downbeat_log_prob.data_ptr<float>();
        float *nonbeat_log_prob_ptr = nonbeat_log_prob.data_ptr<float>();
        int64_t *lengths_ptr = lengths.data_ptr<int64_t>();
        float *score_ptr = score.data_ptr<float>();
        float *best_score_ptr = best_score.data_ptr<float>();
        int64_t *best_prev_ptr = best_prev_states.data_ptr<int64_t>();
        int8_t *peaks_ptr = peaks.data_ptr<int8_t>();

        torch::parallel_for(
            0, batch_size, grain_size,
            [&](int64_t start, int64_t end)
            {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
                {
                    float *_beat_log_prob_ptr = beat_log_prob_ptr + batch_idx * num_frames;
                    float *_downbeat_log_prob_ptr = downbeat_log_prob_ptr + batch_idx * num_frames;
                    float *_nonbeat_log_prob_ptr = nonbeat_log_prob_ptr + batch_idx * num_frames;
                    int64_t _num_frames = lengths_ptr[batch_idx]; // actual number of frames for sample
                    float *_score_ptr = score_ptr + batch_idx * num_frames * num_states;
                    float *_best_score_ptr = best_score_ptr + batch_idx;
                    int64_t *_best_prev_ptr = best_prev_ptr + batch_idx * num_frames * num_states;
                    int8_t *_peaks_ptr = peaks_ptr + batch_idx * num_frames;

                    unbatched_decode(
                        _beat_log_prob_ptr,
                        _downbeat_log_prob_ptr,
                        _nonbeat_log_prob_ptr,
                        _score_ptr,
                        _best_score_ptr,
                        _best_prev_ptr,
                        _peaks_ptr,
                        _num_frames,
                        num_states,
                        num_fpbs,
                        meter,
                        static_cast<float>(beat_region),
                        offsets_ptr,
                        fpbs_ptr,
                        log_transition_prob_ptr);
                }
            });

        return {peaks, best_score};
    }
} // namespace maddad

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY_FRAGMENT(maddad, m)
{
    m.def("decode_beat_and_downbeat_peaks_by_viterbi(Tensor beat_log_prob, Tensor downbeat_log_prob, Tensor nonbeat_log_prob, Tensor lengths, Tensor fpbs, int meter, float beat_region, Tensor log_transition_prob) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(maddad, CPU, m)
{
    m.impl("decode_beat_and_downbeat_peaks_by_viterbi", &maddad::decode_beat_and_downbeat_peaks_by_viterbi);
}
