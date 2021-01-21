import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K


class GRUSkipCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GRUSkipCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        skip_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            initializer="uniform",
            name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            initializer="uniform",
            name="recurrent_kernel")
        self.skip_kernel = self.add_weight(
            shape=(skip_dim, self.units * 3),
            initializer="uniform",
            name="skip_kernel")
        self.bias = self.add_weight(
            shape=(3, 3 * self.units),
            initializer="uniform",
            name="bias")

        self.built = True
    
    def call(self, inputs, states, training=None):
        inputs_item, inputs_skip = inputs
        h_tm1 = states[0]
        input_bias, recurrent_bias, skip_bias = tf.unstack(self.bias)

        matrix_x = K.dot(inputs_item, self.kernel)
        matrix_x = K.bias_add(matrix_x, input_bias)
        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [self.units, self.units, -1], axis=-1)

        matrix_skip = K.dot(inputs_skip, self.skip_kernel)
        matrix_skip = K.bias_add(matrix_skip, skip_bias)
        skip_z, skip_r, skip_h = tf.split(matrix_skip, 3, axis=-1)
        
        z = tf.keras.activations.hard_sigmoid(x_z + skip_z + recurrent_z)
        r = tf.keras.activations.hard_sigmoid(x_r + skip_r + recurrent_r)
        recurrent_h = r * recurrent_h
        hh = tf.keras.activations.tanh(x_h + skip_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh

        return h, h
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError(
                'batch_size and dtype cannot be None while constructing initial state: '
                'batch_size={}, dtype={}'.format(batch_size, dtype))

        flat_dims = tf.TensorShape(self.state_size).as_list()
        init_state_size = [batch_size] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    def get_config(self):
        return {"units" : self.units}


class STABR_song_encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size=64, song_embedding_size=50, name="STABR_song_encoder", **kwargs):
        super(STABR_song_encoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_size, song_embedding_size, mask_zero=True)
        self.gru = layers.Bidirectional(layers.RNN(GRUSkipCell(hidden_size), return_sequences=True, return_state=True))
        self.key_layer = layers.Dense(hidden_size)
        self.query_layer = layers.Dense(hidden_size)
        self.energy_layer = layers.Dense(1)

    def call(self, input_tracks, embedded_skips):
        embedded = self.embedding(input_tracks)
        mask = self.embedding.compute_mask(input_tracks)
        gru_out = self.gru((embedded, embedded_skips), mask=mask)
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
        self.gru = layers.Bidirectional(layers.RNN(GRUSkipCell(hidden_size), return_sequences=True, return_state=True))
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
        songs_context = self.song_encoder(input_songs, embedded_skips)
        tags_context = self.tag_encoder(input_tags, embedded_skips)
        res_logits = self.stabr_forward(songs_context, tags_context, training)

        return res_logits