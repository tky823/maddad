#ifndef MADDAD_H
#define MADDAD_H

#include <torch/extension.h>

namespace maddad
{
    at::Tensor decode_beat_peaks_by_viterbi(const at::Tensor &logit, const at::Tensor &fpbs, const at::Tensor &log_transition_prob);
}

#endif // MADDAD_H
