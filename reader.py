import os

import tensorflow as tf

from common import Common

TARGET_INDEX_KEY = 'TARGET_INDEX_KEY'
TARGET_STRING_KEY = 'TARGET_STRING_KEY'
TARGET_LENGTH_KEY = 'TARGET_LENGTH_KEY'
PATH_SOURCE_INDICES_KEY = 'PATH_SOURCE_INDICES_KEY'
NODE_INDICES_KEY = 'NODES_INDICES_KEY'
PATH_TARGET_INDICES_KEY = 'PATH_TARGET_INDICES_KEY'
VALID_CONTEXT_MASK_KEY = 'VALID_CONTEXT_MASK_KEY'
PATH_SOURCE_LENGTHS_KEY = 'PATH_SOURCE_LENGTHS_KEY'
PATH_LENGTHS_KEY = 'PATH_LENGTHS_KEY'
PATH_TARGET_LENGTHS_KEY = 'PATH_TARGET_LENGTHS_KEY'
PATH_SOURCE_STRINGS_KEY = 'PATH_SOURCE_STRINGS_KEY'
PATH_STRINGS_KEY = 'PATH_STRINGS_KEY'
PATH_TARGET_STRINGS_KEY = 'PATH_TARGET_STRINGS_KEY'

NODE_IDS_KEY = 'NODE_IDS'
SUBTOKEN_IDS_KEY = 'SUBTOKEN_IDS'
TARGET_IDS_KEY = 'TARGET_IDS'
PARENT_INDICES_KEY = 'PARENT_INDICES'
INCIDENCE_MATRIX_KEY = 'INCIDENCE_MATRIX'

def scalar_string_split(input, delimiter):
    return tf.reshape(
        tf.sparse.to_dense(
            tf.string_split(tf.expand_dims(input, -1), delimiter, skip_empty=False),
            default_value=Common.PAD,
        ),
        shape=[-1],
    )

def pad_up_to(input, target_dims, constant_values):
    shape = tf.shape(input)
    input = tf.slice(input, (0,) * len(target_dims), tf.minimum(shape, target_dims)) # range-safe slice
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
        parent_indices = pad_up_to(parent_indices, (MAX_NODES,), -1)

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
            NODE_IDS_KEY: node_ids,
            SUBTOKEN_IDS_KEY: nodes_subtokens_ids,
            TARGET_IDS_KEY: target_ids,
            PARENT_INDICES_KEY: parent_indices,
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


if __name__ == '__main__':
    target_word_to_index = {Common.PAD: 0, Common.UNK: 1, Common.SOS: 2,
                            'a': 3, 'b': 4, 'c': 5, 'd': 6, 't': 7}
    subtoken_to_index = {Common.PAD: 0, Common.UNK: 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    node_to_index = {Common.PAD: 0, Common.UNK: 1, '1': 2, '2': 3, '3': 4, '4': 5}
    import numpy as np


    class Config:
        def __init__(self):
            self.SAVE_EVERY_EPOCHS = 1
            self.TRAIN_PATH = self.TEST_PATH = 'test_input/test_input'
            self.BATCH_SIZE = 2
            self.TEST_BATCH_SIZE = self.BATCH_SIZE
            self.READER_NUM_PARALLEL_BATCHES = 1
            self.READING_BATCH_SIZE = 2
            self.SHUFFLE_BUFFER_SIZE = 100
            self.MAX_CONTEXTS = 4
            self.DATA_NUM_CONTEXTS = 4
            self.MAX_PATH_LENGTH = 3
            self.MAX_NAME_PARTS = 2
            self.MAX_TARGET_PARTS = 4
            self.RANDOM_CONTEXTS = True
            self.CSV_BUFFER_SIZE = None


    config = Config()
    reader = Reader(subtoken_to_index, target_word_to_index, node_to_index, config, False)

    output = reader.get_output()
    target_index_op = output[TARGET_INDEX_KEY]
    target_string_op = output[TARGET_STRING_KEY]
    target_length_op = output[TARGET_LENGTH_KEY]
    path_source_indices_op = output[PATH_SOURCE_INDICES_KEY]
    node_indices_op = output[NODE_INDICES_KEY]
    path_target_indices_op = output[PATH_TARGET_INDICES_KEY]
    valid_context_mask_op = output[VALID_CONTEXT_MASK_KEY]
    path_source_lengths_op = output[PATH_SOURCE_LENGTHS_KEY]
    path_lengths_op = output[PATH_LENGTHS_KEY]
    path_target_lengths_op = output[PATH_TARGET_LENGTHS_KEY]
    path_source_strings_op = output[PATH_SOURCE_STRINGS_KEY]
    path_strings_op = output[PATH_STRINGS_KEY]
    path_target_strings_op = output[PATH_TARGET_STRINGS_KEY]

    sess = tf.InteractiveSession()
    tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
    reader.reset(sess)

    try:
        while True:
            target_indices, target_strings, target_lengths, path_source_indices, \
            node_indices, path_target_indices, valid_context_mask, path_source_lengths, \
            path_lengths, path_target_lengths, path_source_strings, path_strings, \
            path_target_strings = sess.run(
                [target_index_op, target_string_op, target_length_op, path_source_indices_op,
                 node_indices_op, path_target_indices_op, valid_context_mask_op, path_source_lengths_op,
                 path_lengths_op, path_target_lengths_op, path_source_strings_op, path_strings_op,
                 path_target_strings_op])

            print('Target strings: ', Common.binary_to_string_list(target_strings))
            print('Context strings: ', Common.binary_to_string_3d(
                np.concatenate([path_source_strings, path_strings, path_target_strings], -1)))
            print('Target indices: ', target_indices)
            print('Target lengths: ', target_lengths)
            print('Path source strings: ', Common.binary_to_string_3d(path_source_strings))
            print('Path source indices: ', path_source_indices)
            print('Path source lengths: ', path_source_lengths)
            print('Path strings: ', Common.binary_to_string_3d(path_strings))
            print('Node indices: ', node_indices)
            print('Path lengths: ', path_lengths)
            print('Path target strings: ', Common.binary_to_string_3d(path_target_strings))
            print('Path target indices: ', path_target_indices)
            print('Path target lengths: ', path_target_lengths)
            print('Valid context mask: ', valid_context_mask)

    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')
