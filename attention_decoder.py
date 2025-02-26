# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file defines the decoder with attention mechanism for TensorFlow 2.x."""

import tensorflow as tf
import logging

# Use logging instead of tf.logging
logging.basicConfig(level=logging.INFO)

class AttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, cell, attn_size, pointer_gen=True, use_coverage=False):
        super(AttentionDecoder, self).__init__()
        self.cell = cell  # LSTM cell
        self.attn_size = attn_size
        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage

        # Define layers
        self.W_h = tf.Variable(tf.random.normal([1, 1, attn_size, attn_size]), name="W_h")
        self.v = tf.Variable(tf.random.normal([attn_size]), name="v")
        if self.use_coverage:
            self.w_c = tf.Variable(tf.random.normal([1, 1, 1, attn_size]), name="w_c")

    def attention(self, decoder_state, encoder_states, enc_padding_mask, coverage=None):
        """Calculate attention distribution and context vector."""
        decoder_features = tf.keras.layers.Dense(self.attn_size, activation="tanh")(decoder_state)
        decoder_features = tf.reshape(decoder_features, [-1, 1, 1, self.attn_size])

        def masked_attention(e):
            """Apply softmax and mask out padding tokens."""
            attn_dist = tf.nn.softmax(e)
            attn_dist *= enc_padding_mask  # Apply padding mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1, keepdims=True)
            return attn_dist / masked_sums  # Normalize

        if self.use_coverage and coverage is not None:
            coverage_features = tf.nn.conv2d(coverage, self.w_c, [1, 1, 1, 1], "SAME")
            e = tf.reduce_sum(self.v * tf.tanh(encoder_states + decoder_features + coverage_features), [2, 3])
            attn_dist = masked_attention(e)
            coverage += tf.reshape(attn_dist, [tf.shape(encoder_states)[0], -1, 1, 1])
        else:
            e = tf.reduce_sum(self.v * tf.tanh(encoder_states + decoder_features), [2, 3])
            attn_dist = masked_attention(e)
            if self.use_coverage:
                coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)

        context_vector = tf.reduce_sum(tf.reshape(attn_dist, [-1, tf.shape(encoder_states)[1], 1, 1]) * encoder_states, [1, 2])
        context_vector = tf.reshape(context_vector, [-1, self.attn_size])

        return context_vector, attn_dist, coverage

    def call(self, decoder_inputs, initial_state, encoder_states, enc_padding_mask):
        """Run the attention decoder."""
        outputs, attn_dists, p_gens = [], [], []
        state = initial_state
        coverage = None
        context_vector = tf.zeros([tf.shape(encoder_states)[0], self.attn_size])

        for i, inp in enumerate(decoder_inputs):
            logging.info(f"Adding attention_decoder timestep {i} of {len(decoder_inputs)}")
            x = tf.keras.layers.Dense(inp.shape[-1], activation="relu")(tf.concat([inp, context_vector], axis=-1))

            cell_output, state = self.cell(x, states=state)
            context_vector, attn_dist, coverage = self.attention(state, encoder_states, enc_padding_mask, coverage)
            attn_dists.append(attn_dist)

            if self.pointer_gen:
                p_gen = tf.keras.layers.Dense(1, activation="sigmoid")(tf.concat([context_vector, state[0], state[1], x], axis=-1))
                p_gens.append(p_gen)

            output = tf.keras.layers.Dense(self.cell.output_size)(tf.concat([cell_output, context_vector], axis=-1))
            outputs.append(output)

        return outputs, state, attn_dists, p_gens, coverage
