import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as initializers


class LSTMSkipCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(LSTMSkipCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units]
        self.output_size = self.units
    
    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        skip_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer=initializers.get("glorot_uniform"),
            name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer=initializers.get("orthogonal"),
            name="recurrent_kernel")
        self.skip_kernel = self.add_weight(
            shape=(skip_dim, self.units * 4),
            initializer=initializers.get("glorot_uniform"),
            name="skip_kernel")
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer=initializers.get("zeros"),
            name="bias")
        
        self.built = True
    
    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = tf.keras.activations.sigmoid(z0)
        f = tf.keras.activations.sigmoid(z1)
        c = f * c_tm1 + i * tf.keras.activations.tanh(z2)
        o = tf.keras.activations.sigmoid(z3)

        return c, o

    def call(self, inputs, states, training=None):
        inputs_item, inputs_skip = inputs
        h_tm1 = states[0]
        c_tm1 = states[1]

        z = K.dot(inputs_item, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)
        z += K.dot(inputs_skip, self.skip_kernel)
        z = K.bias_add(z, self.bias)
        z = tf.split(z, num_or_size_splits=4, axis=1)

        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        h = o * tf.keras.activations.tanh(c)
        
        return h, [h, c]
    
    def get_config(self):
        return {"units" : self.units}


class STABR_song_encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size=64, song_embedding_size=50, name="STABR_song_encoder", **kwargs):
        super(STABR_song_encoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_size, song_embedding_size, mask_zero=True)
        self.gru = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True, return_state=True))
        self.key_layer = layers.Dense(hidden_size)
        self.query_layer = layers.Dense(hidden_size)
        self.energy_layer = layers.Dense(1)

    def call(self, input_tracks):
        embedded = self.embedding(input_tracks)
        mask = self.embedding.compute_mask(input_tracks)
        gru_out = self.gru(embedded, mask=mask)
        query_with_time_axis = tf.expand_dims(gru_out[1], 1)
        keys = self.key_layer(gru_out[0])
        queries = self.query_layer(query_with_time_axis)
        attn_applied = self.energy_layer(tf.nn.tanh(keys + queries))
        softmax_mask = tf.cast(mask, tf.float32)
        softmax_mask = 1 - softmax_mask
        softmax_mask = tf.expand_dims(softmax_mask, 2)
        attn_applied += softmax_mask * -1e9
        weights = tf.nn.softmax(attn_applied, axis=1)
        context = weights * gru_out[0]
        context = tf.reduce_sum(context, axis=1)

        return context


class STABR_tag_encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size=64, tag_embedding_size=25, name="STABR_tag_encoder", **kwargs):
        super(STABR_tag_encoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_size, tag_embedding_size, mask_zero=True)
        self.gru = layers.Bidirectional(layers.RNN(LSTMSkipCell(hidden_size), return_sequences=True, return_state=True))
        self.key_layer = layers.Dense(hidden_size)
        self.query_layer = layers.Dense(hidden_size)
        self.energy_layer = layers.Dense(1)

    def call(self, input_tags, embedded_skips):
        embedded = self.embedding(input_tags)
        mask = self.embedding.compute_mask(input_tags)
        num_mask = tf.cast(mask, tf.float32)
        masked_embedded = embedded * tf.expand_dims(num_mask, 3)
        denom = tf.reduce_sum(num_mask, axis=2)
        denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom, dtype=tf.float32), denom)
        gru_inp = tf.reduce_sum(masked_embedded, axis=2) / tf.expand_dims(denom, 2)
        gru_out = self.gru((gru_inp, embedded_skips), mask=mask[:,:,0])
        query_with_time_axis = tf.expand_dims(gru_out[1], 1)
        keys = self.key_layer(gru_out[0])
        queries = self.query_layer(query_with_time_axis)
        attn_applied = self.energy_layer(tf.nn.tanh(keys + queries))
        softmax_mask = tf.cast(mask[:,:,0], tf.float32)
        softmax_mask = 1 - softmax_mask
        softmax_mask = tf.expand_dims(softmax_mask, 2)
        attn_applied += softmax_mask * -1e9
        weights = tf.nn.softmax(attn_applied, axis=1)
        context = weights * gru_out[0]
        context = tf.reduce_sum(context, axis=1)

        return context


class STABR_forward(tf.keras.layers.Layer):
    def __init__(self, total_songs, dropout=0.1, v_layer_size=50, name="STABR_forward", **kwargs):
        super(STABR_forward, self).__init__(name=name, **kwargs)
        self.v_layer = layers.Dense(v_layer_size, activation="relu")
        self.out_layer = layers.Dense(total_songs)
        self.drop_rate = dropout
    
    def call(self, songs_context, tags_context, training=True):
        context_c = tf.concat([songs_context, tags_context], 1)
        vec_rep = self.v_layer(context_c)
        if training:
            vec_rep = tf.nn.dropout(vec_rep, rate=self.drop_rate)
        res = self.out_layer(vec_rep)

        return res


class STABR(tf.keras.Model):
    def __init__(self, song_input_size, song_embedding_size,
                 tag_input_size, tag_embedding_size,
                 skips_input_size, skips_embedding_size, hidden_size=64,
                 dropout=0.1, v_layer_size=50, name="STABR", **kwargs):
        super(STABR, self).__init__(name=name, **kwargs)
        self.song_encoder = STABR_song_encoder(song_input_size, hidden_size=hidden_size, song_embedding_size=song_embedding_size)
        self.tag_encoder = STABR_tag_encoder(tag_input_size, hidden_size=hidden_size, tag_embedding_size=tag_embedding_size)
        self.stabr_forward = STABR_forward(song_input_size, dropout=dropout, v_layer_size=v_layer_size)
        self.skips_embedding = layers.Embedding(skips_input_size, skips_embedding_size, mask_zero=True)
    
    def call(self, inp, training=None):
        input_songs, input_tags, input_skips = inp
        embedded_skips = self.skips_embedding(input_skips)
        songs_context = self.song_encoder(input_songs)
        tags_context = self.tag_encoder(input_tags, embedded_skips)
        res_logits = self.stabr_forward(songs_context, tags_context, training)

        return res_logits
