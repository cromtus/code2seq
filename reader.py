import os

import tensorflow as tf

from common import Common


TARGET_STRING_KEY = 'TARGET_STRING'
NODE_IDS_KEY = 'NODE_IDS'
SUBTOKEN_IDS_KEY = 'SUBTOKEN_IDS'
TARGET_IDS_KEY = 'TARGET_IDS'
INCIDENCE_MATRIX_KEY = 'INCIDENCE_MATRIX'

def scalar_string_split(input, delimiter):
    return tf.reshape(
        tf.sparse.to_dense(
            tf.string_split(tf.expand_dims(input, -1), delimiter, skip_empty=False),
            default_value=Common.PAD,
        ),
        shape=[-1],
    )

def crop_to(input, target_dims):
    shape = tf.shape(input)
    return tf.slice(input, (0,) * len(target_dims), tf.minimum(shape, target_dims)) # range-safe slice

def pad_up_to(input, target_dims, constant_values):
    input = crop_to(input, target_dims)
    shape = tf.shape(input)
    paddings = [[0, m - shape[i]] for i, m in enumerate(target_dims)]
    return tf.pad(input, paddings, 'CONSTANT', constant_values=constant_values)

class Reader:
    class_subtoken_table = None
    class_target_table = None
    class_node_table = None

    def __init__(self, subtoken_to_index, target_to_index, node_to_index, config, is_evaluating=False):
        self.config = config
        self.file_path = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train.c2s')
        if self.file_path is not None and not os.path.exists(self.file_path):
            print(
                '%s cannot find file: %s' % ('Evaluation reader' if is_evaluating else 'Train reader', self.file_path))
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE
        self.is_evaluating = is_evaluating

        self.context_pad = '{},{},{}'.format(Common.PAD, Common.PAD, Common.PAD)
        self.record_defaults = [[self.context_pad]] * 4

        self.subtoken_table = Reader.get_subtoken_table(subtoken_to_index)
        self.target_table = Reader.get_target_table(target_to_index)
        self.node_table = Reader.get_node_table(node_to_index)
        self.SOS_id = target_to_index[Common.SOS]
        self.EOS_id = target_to_index[Common.EOS]
        if self.file_path is not None:
            self.output_tensors = self.compute_output()

    @classmethod
    def get_subtoken_table(cls, subtoken_to_index):
        if cls.class_subtoken_table is None:
            cls.class_subtoken_table = cls.initialize_hash_map(subtoken_to_index, subtoken_to_index[Common.UNK])
        return cls.class_subtoken_table

    @classmethod
    def get_target_table(cls, target_to_index):
        if cls.class_target_table is None:
            cls.class_target_table = cls.initialize_hash_map(target_to_index, target_to_index[Common.UNK])
        return cls.class_target_table

    @classmethod
    def get_node_table(cls, node_to_index):
        if cls.class_node_table is None:
            cls.class_node_table = cls.initialize_hash_map(node_to_index, node_to_index[Common.UNK])
        return cls.class_node_table

    @classmethod
    def initialize_hash_map(cls, word_to_index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32), default_value)

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, record_defaults=self.record_defaults, field_delim=' ', use_quote_delim=False)
        return self.process_dataset(*parts)

    def process_dataset(self, *row_parts):
        MAX_NODES = self.config.MAX_CONTEXTS
        row_parts = list(row_parts)

        target_subtokens = scalar_string_split(row_parts[0], delimiter='|') # (n_target_tokens,)
        target_subtokens = pad_up_to(target_subtokens, (self.config.MAX_TARGET_PARTS,), Common.PAD)
        target_ids = self.target_table.lookup(target_subtokens)

        node_types = scalar_string_split(row_parts[1], delimiter=',') # (n_nodes,)
        node_types = pad_up_to(node_types, (MAX_NODES,), Common.PAD)
        node_ids = self.node_table.lookup(node_types)

        node_names = scalar_string_split(row_parts[2], delimiter=',') # (n_nodes,)
        nodes_subtokens_sparse = tf.string_split(node_names, delimiter='|') # (n_nodes, n_tokens_i)
        nodes_subtokens = tf.sparse.to_dense(nodes_subtokens_sparse, Common.PAD) # (n_nodes, max(n_tokens))
        nodes_subtokens = pad_up_to(nodes_subtokens, (MAX_NODES, self.config.MAX_NAME_PARTS), Common.PAD)
        nodes_subtokens_ids = self.subtoken_table.lookup(nodes_subtokens)

        parent_indices = scalar_string_split(row_parts[3], delimiter=',') # (n_nodes,)
        parent_indices = tf.strings.to_number(parent_indices, tf.int32)
        parent_indices = crop_to(parent_indices, (MAX_NODES,))

        children_indices = tf.range(tf.shape(parent_indices)[0])
        incident_indices = tf.transpose([children_indices, parent_indices])[1:] # 0th node has no parent
        incident_indices = tf.concat([incident_indices, incident_indices[:, ::-1]], axis=0) # symmetry
        incidence_matrix = tf.SparseTensor(
            tf.cast(incident_indices, tf.int64),
            tf.ones(tf.shape(incident_indices)[0]),
            (MAX_NODES, MAX_NODES),
        )
        incidence_matrix = tf.sparse.reorder(incidence_matrix)
        incidence_matrix = tf.sparse.to_dense(incidence_matrix)

        return {
            TARGET_STRING_KEY: row_parts[0],
            NODE_IDS_KEY: node_ids,
            SUBTOKEN_IDS_KEY: nodes_subtokens_ids,
            TARGET_IDS_KEY: target_ids,
            INCIDENCE_MATRIX_KEY: incidence_matrix,
        }

    def reset(self, sess):
        sess.run(self.reset_op)

    def get_output(self):
        return self.output_tensors

    def compute_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=self.record_defaults, field_delim=' ',
                                                  use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=self.batch_size,
            num_parallel_batches=self.config.READER_NUM_PARALLEL_BATCHES))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self.iterator = dataset.make_initializable_iterator()
        self.reset_op = self.iterator.initializer
        return self.iterator.get_next()
