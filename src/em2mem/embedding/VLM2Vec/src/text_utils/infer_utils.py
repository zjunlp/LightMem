# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch

def hfmodel_generate(tokenizer, model, prompt, generate_kwargs={}):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded_input.input_ids
    assert len(input_ids) == 1, len(input_ids)
    prompt_len = input_ids.shape[1]
    output = model.generate(
        **encoded_input,
        pad_token_id=tokenizer.eos_token_id,
        **generate_kwargs
    )
    output_ids = output.sequences[:, prompt_len + 3:]
    output_seq = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)[0].strip()
    output_tokens = [tokenizer.decode(t) for t in output_ids.cpu()]
    probs = torch.nn.functional.softmax(torch.cat(output.scores, dim=0), dim=-1)

    return output_seq
