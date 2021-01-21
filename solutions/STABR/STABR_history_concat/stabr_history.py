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
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            initializer="uniform",
            name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            initializer="uniform",
            name="recurrent_kernel")
        self.bias = self.add_weight(
            shape=(2, 3 * self.units),
            initializer="uniform",
            name="bias")

        self.built = True
    
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]
        input_bias, recurrent_bias = tf.unstack(self.bias)
        matrix_x = K.dot(inputs, self.kernel)
        matrix_x = K.bias_add(matrix_x, input_bias)
        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [self.units, self.units, -1], axis=-1)
        
        z = tf.keras.activations.hard_sigmoid(x_z + recurrent_z)
        r = tf.keras.activations.hard_sigmoid(x_r + recurrent_r)
        recurrent_h = r * recurrent_h
        hh = tf.keras.activations.tanh(x_h + recurrent_h)
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


class SongHistoryEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name="song_history_encoder", **kwargs):
        super(SongHistoryEncoder, self).__init__(name=name, **kwargs)
        self.gru = layers.GRU(hidden_size)
    
    def call(self, input_batch, mask):
        mask_num = tf.cast(mask, tf.float32)
        input_batch = input_batch * tf.expand_dims(mask_num, 3)
        denom = tf.reduce_sum(mask_num, axis=2)
        denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom, dtype=tf.float32), denom)
        averaged = tf.reduce_sum(input_batch, axis=2) / tf.expand_dims(denom, 2)
        out = self.gru(averaged, mask=mask[:,:,0])

        return out


class STABR_song_encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size=64, song_embedding_size=50,
                 track_history_hidden_size=64, name="STABR_song_encoder", **kwargs):
        super(STABR_song_encoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_size, song_embedding_size, mask_zero=True)
        self.gru = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True, return_state=True))
        self.key_layer = layers.Dense(hidden_size)
        self.query_layer = layers.Dense(hidden_size)
        self.energy_layer = layers.Dense(1)
        self.history_encoder = SongHistoryEncoder(track_history_hidden_size)

    def call(self, input_batch, input_history):
        embedded = self.embedding(input_batch)
        embedded_history = self.embedding(input_history)
        mask = self.embedding.compute_mask(input_batch)
        mask_history = self.embedding.compute_mask(input_history)
        gru_out = self.gru(embedded, mask=mask)
        history = self.history_encoder(embedded_history, mask_history)
        pre_query = tf.concat([gru_out[1], history], 1)
        pre_query = tf.expand_dims(pre_query, 1) # expand history with time axis
        keys = self.key_layer(gru_out[0])
        queries = self.query_layer(pre_query)
        attn_applied = self.energy_layer(tf.nn.tanh(keys + queries))
        softmax_mask = tf.cast(mask, tf.float32)
        softmax_mask = 1 - softmax_mask
        softmax_mask = tf.expand_dims(softmax_mask, 2)
        attn_applied += softmax_mask * -1e9
        weights = tf.nn.softmax(attn_applied, axis=1)
        context = weights * gru_out[0]
        context = tf.reduce_sum(context, axis=1)

        return context


class TagHistoryEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, name="tag_history_encoder", **kwargs):
        super(TagHistoryEncoder, self).__init__(name=name, **kwargs)
        self.gru = layers.GRU(hidden_size)
    
    def call(self, input_batch, mask):
        mask_num = tf.cast(mask, tf.float32)
        input_batch = input_batch * tf.expand_dims(mask_num, 4)
        denom = tf.reduce_sum(mask_num, axis=3)
        denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom, dtype=tf.float32), denom)
        averaged = tf.reduce_sum(input_batch, axis=3) / tf.expand_dims(denom, 3)
        new_mask = mask_num[:,:,:,0]
        average_masked = averaged * tf.expand_dims(new_mask, 3)
        new_denom = tf.reduce_sum(new_mask, axis=2)
        new_denom = tf.where(tf.equal(new_denom, 0), tf.ones_like(new_denom, dtype=tf.float32), new_denom)
        new_averaged = tf.reduce_sum(average_masked, axis=2) / tf.expand_dims(new_denom, 2)
        out = self.gru(new_averaged, mask=mask[:,:,:,0][:,:,0])

        return out


class STABR_tag_encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size=64, tag_embedding_size=25,
                 tag_history_hidden_size=64, name="STABR_tag_encoder", **kwargs):
        super(STABR_tag_encoder, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_size, tag_embedding_size, mask_zero=True)
        self.gru = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True, return_state=True))
        self.key_layer = layers.Dense(hidden_size)
        self.query_layer = layers.Dense(hidden_size)
        self.energy_layer = layers.Dense(1)
        self.history_encoder = TagHistoryEncoder(tag_history_hidden_size)

    def call(self, input_batch, input_history):
        embedded = self.embedding(input_batch)
        embedded_history = self.embedding(input_history)
        mask = self.embedding.compute_mask(input_batch)
        mask_history = self.embedding.compute_mask(input_history)
        num_mask = tf.cast(mask, tf.float32)
        masked_embedded = embedded * tf.expand_dims(num_mask, 3)
        denom = tf.reduce_sum(num_mask, axis=2)
        denom = tf.where(tf.equal(denom, 0), tf.ones_like(denom, dtype=tf.float32), denom)
        gru_inp = tf.reduce_sum(masked_embedded, axis=2) / tf.expand_dims(denom, 2)
        gru_out = self.gru(gru_inp, mask=mask[:,:,0])
        history = self.history_encoder(embedded_history, mask_history)
        pre_query = tf.concat([gru_out[1], history], 1)
        pre_query = tf.expand_dims(pre_query, 1) # expand history with time axis
        keys = self.key_layer(gru_out[0])
        queries = self.query_layer(pre_query)
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
                 tag_input_size, tag_embedding_size, hidden_size=64,
                 track_history_hidden_size=64, tag_history_hidden_size=64,
                 dropout=0.1, v_layer_size=50, name="STABR", **kwargs):
        super(STABR, self).__init__(name=name, **kwargs)
        self.song_encoder = STABR_song_encoder(song_input_size, hidden_size=hidden_size,
            song_embedding_size=song_embedding_size, track_history_hidden_size=track_history_hidden_size)
        self.tag_encoder = STABR_tag_encoder(tag_input_size, hidden_size=hidden_size,
            tag_embedding_size=tag_embedding_size, tag_history_hidden_size=tag_history_hidden_size)
        self.stabr_forward = STABR_forward(song_input_size, dropout=dropout, v_layer_size=v_layer_size)
    
    def call(self, inp, training=True):
        input_songs, songs_history, input_tags, tags_history = inp
        songs_context = self.song_encoder(input_songs, songs_history)
        tags_context = self.tag_encoder(input_tags, tags_history)
        res_logits = self.stabr_forward(songs_context, tags_context, training)

        return res_logits
