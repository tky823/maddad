#!/bin/bash

set -eu
set -o pipefail

data_root="../data"

unpack=true  # unpack .tar.gz or not
chunk_size=8192  # chunk size in byte to download

. ../../_common/parse_options.sh || exit 1;

maddad-download-beatthis \
root="${data_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
