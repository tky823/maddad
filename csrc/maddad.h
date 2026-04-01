#ifndef MADDAD_H
#define MADDAD_H

#include <torch/extension.h>

namespace maddad
{
    at::Tensor decode_beat_peaks_by_viterbi(const at::Tensor &beat_log_prob, const at::Tensor &nonbeat_log_prob, const at::Tensor &lengths, const at::Tensor &fpbs, const at::Tensor &log_transition_prob);
    std::tuple<at::Tensor, at::Tensor> decode_beat_and_downbeat_peaks_by_viterbi(const at::Tensor &beat_log_prob, const at::Tensor &downbeat_log_prob, const at::Tensor &nonbeat_log_prob, const at::Tensor &lengths, const at::Tensor &fpbs, const int8_t meters, const at::Tensor &log_transition_prob);
}

#endif // MADDAD_H
