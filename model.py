import _pickle as pickle
import os
import time

import numpy as np
import shutil
import tensorflow as tf

import reader
from common import Common
from rouge import FilesRouge

from transformer import Encoder, Decoder

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

class Model:
    topk = 10
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_predicted_indices_op, self.eval_top_values_op, self.eval_true_target_strings_op = None, None, None
        self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op = None, None, None
        self.subtoken_to_index = None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')

            self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                            max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS, Common.EOS],
                                            max_size=config.TARGET_VOCAB_MAX_SIZE)
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

            self.encoder = Encoder(self.config.NUM_ENCODER_LAYERS, self.config.EMBEDDINGS_SIZE * 2, self.config.NUM_ATTENTION_HEADS, 512)
            self.decoder = Decoder(self.config.NUM_DECODER_LAYERS, self.config.EMBEDDINGS_SIZE, self.config.NUM_ATTENTION_HEADS, 512,
                            self.target_vocab_size, self.config.MAX_TARGET_PARTS)
            self.epochs_trained = 0

    def close_session(self):
        self.sess.close()

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        best_f1 = 0
        best_epoch = 0
        best_f1_precision = 0
        best_f1_recall = 0
        epochs_no_improve = 0

        self.queue_thread = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                          node_to_index=self.node_to_index,
                                          target_to_index=self.target_to_index,
                                          config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())
        self.print_hyperparams()
        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        time.sleep(1)
        print('Started reader...')

        multi_batch_start_time = time.time()
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            self.queue_thread.reset(self.sess)
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])
                    sum_loss += batch_loss
                    # print('SINGLE BATCH LOSS', batch_loss)
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        sum_loss = 0
                        multi_batch_start_time = time.time()


            except tf.errors.OutOfRangeError:
                self.epochs_trained += self.config.SAVE_EVERY_EPOCHS
                print('Finished %d epochs' % self.config.SAVE_EVERY_EPOCHS)
                results, precision, recall, f1, rouge = self.evaluate()
                print('Accuracy after %d epochs: %.5f' % (self.epochs_trained, results))
                print('After %d epochs: Precision: %.5f, recall: %.5f, F1: %.5f' % (
                    self.epochs_trained, precision, recall, f1))
                print('Rouge: ', rouge)
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_precision = precision
                    best_f1_recall = recall
                    best_epoch = self.epochs_trained
                    epochs_no_improve = 0
                    self.save_model(self.sess, self.config.SAVE_PATH)
                else:
                    epochs_no_improve += self.config.SAVE_EVERY_EPOCHS
                    if epochs_no_improve >= self.config.PATIENCE:
                        print('Not improved for %d epochs, stopping training' % self.config.PATIENCE)
                        print('Best scores - epoch %d: ' % best_epoch)
                        print('Precision: %.5f, recall: %.5f, F1: %.5f' % (best_f1_precision, best_f1_recall, best_f1))
                        return

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH + '.final')
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sh%sm%ss\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / self.num_batches_to_log
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    def evaluate(self, release=False):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                            node_to_index=self.node_to_index,
                                            target_to_index=self.target_to_index,
                                            config=self.config, is_evaluating=True)
            reader_output = self.eval_queue.get_output()
            self.eval_predicted_indices_op = self.build_test_graph(reader_output)
            self.eval_true_target_strings_op = reader_output[reader.TARGET_STRING_KEY]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if release:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name)
                self.saver.save(self.sess, release_name)
                shutil.copyfile(src=self.config.LOAD_PATH + '.dict', dst=release_name + '.dict')
                return None
        model_dirname = os.path.dirname(self.config.SAVE_PATH if self.config.SAVE_PATH else self.config.LOAD_PATH)
        ref_file_name = model_dirname + '/ref.txt'
        predicted_file_name = model_dirname + '/pred.txt'
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)

        with open(model_dirname + '/log.txt', 'w') as output_file, open(ref_file_name, 'w') as ref_file, open(
                predicted_file_name,
                'w') as pred_file:
            num_correct_predictions = 0
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            self.eval_queue.reset(self.sess)
            start_time = time.time()

            try:
                while True:
                    predicted_indices, true_target_strings = self.sess.run(
                        [self.eval_predicted_indices_op, self.eval_true_target_strings_op],
                    )
                    true_target_strings = Common.binary_to_string_list(true_target_strings)
                    ref_file.write(
                        '\n'.join(
                            [name.replace(Common.internal_delimiter, ' ') for name in true_target_strings]) + '\n')
                    predicted_strings = [[self.index_to_target[i] for i in example]
                                            for example in predicted_indices]
                    pred_file.write('\n'.join(
                        [' '.join(Common.filter_impossible_names(words)) for words in predicted_strings]) + '\n')

                    num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                              zip(true_target_strings,
                                                                                  predicted_strings))
                    true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                        zip(true_target_strings, predicted_strings),
                        true_positive, false_positive, false_negative)

                    total_predictions += len(true_target_strings)
                    total_prediction_batches += 1
                    if total_prediction_batches % self.num_batches_to_log == 0:
                        elapsed = time.time() - start_time
                        self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)
            except tf.errors.OutOfRangeError:
                pass

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
            # Common.compute_bleu(ref_file_name, predicted_file_name)

        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        try:
            files_rouge = FilesRouge()
            rouge = files_rouge.get_scores(
                hyp_path=predicted_file_name, ref_path=ref_file_name, avg=True, ignore_empty=True)
        except ValueError:
            rouge = 0
        print("Evaluation time: %sh%sm%ss" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return num_correct_predictions / total_predictions, \
               precision, recall, f1, rouge

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, predicted in results:
            original_name_parts = original_name.split(Common.internal_delimiter) # list
            filtered_original = Common.filter_impossible_names(original_name_parts) # list
            predicted_first = predicted
            filtered_predicted_first_parts = Common.filter_impossible_names(predicted_first) # list

            output_file.write('Original: ' + Common.internal_delimiter.join(original_name_parts) +
                                ' , predicted 1st: ' + Common.internal_delimiter.join(filtered_predicted_first_parts) + '\n')
            if filtered_original == filtered_predicted_first_parts or Common.unique(filtered_original) == Common.unique(
                    filtered_predicted_first_parts) or ''.join(filtered_original) == ''.join(filtered_predicted_first_parts):
                num_correct_predictions += 1
        return num_correct_predictions

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, predicted in results:
            filtered_predicted_names = Common.filter_impossible_names(predicted)
            filtered_original_subtokens = Common.filter_impossible_names(original_name.split(Common.internal_delimiter))

            if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
                true_positive += len(filtered_original_subtokens)
                continue

            for subtok in filtered_predicted_names:
                if subtok in filtered_original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in filtered_original_subtokens:
                if not subtok in filtered_predicted_names:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    def print_hyperparams(self):
        print('Training batch size:\t\t\t', self.config.BATCH_SIZE)
        print('Dataset path:\t\t\t\t', self.config.TRAIN_PATH)
        print('Training file path:\t\t\t', self.config.TRAIN_PATH + '.train.c2s')
        print('Validation path:\t\t\t', self.config.TEST_PATH)
        print('Taking max contexts from each example:\t', self.config.MAX_CONTEXTS)
        print('Random path sampling:\t\t\t', self.config.RANDOM_CONTEXTS)
        print('Embedding size:\t\t\t\t', self.config.EMBEDDINGS_SIZE)
        print('Decoder size:\t\t\t\t', self.config.DECODER_SIZE)
        print('Decoder layers:\t\t\t\t', self.config.NUM_DECODER_LAYERS)
        print('Max path lengths:\t\t\t', self.config.MAX_PATH_LENGTH)
        print('Max subtokens in a token:\t\t', self.config.MAX_NAME_PARTS)
        print('Max target length:\t\t\t', self.config.MAX_TARGET_PARTS)
        print('Embeddings dropout keep_prob:\t\t', self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)
        print('LSTM dropout keep_prob:\t\t\t', self.config.RNN_DROPOUT_KEEP_PROB)
        print('============================================')

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed):
        accuracy_message = str(correct_predictions / total_predictions)
        throughput_message = "Prediction throughput: %d" % int(total_predictions / (elapsed if elapsed > 0 else 1))
        output_file.write(accuracy_message + '\n')
        output_file.write(throughput_message)
        # print(accuracy_message)
        print(throughput_message)

    def build_training_graph(self, input_tensors):
        node_ids = input_tensors[reader.NODE_IDS_KEY]
        nodes_subtokens_ids = input_tensors[reader.SUBTOKEN_IDS_KEY]
        output_target_ids = input_tensors[reader.TARGET_IDS_KEY]
        incidence_matrix = input_tensors[reader.INCIDENCE_MATRIX_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))

            batch_size = tf.shape(node_ids)[0]

            input_mask = create_padding_mask(node_ids)
            nodes_emb = self.prepare_node_embeddings(nodes_vocab, node_ids, subtoken_vocab, nodes_subtokens_ids)
            incidence_matrix = incidence_matrix[:, tf.newaxis, :, :]
            encoded = self.encoder(nodes_emb, True, input_mask, incidence_matrix)

            SOS_column = tf.fill((batch_size, 1), self.queue_thread.SOS_id)
            input_target_ids = tf.concat([SOS_column, output_target_ids[:, :-1]], axis=-1) # SOS shift
            target_mask = create_padding_mask(input_target_ids)
            look_ahead_mask = create_look_ahead_mask(self.config.MAX_TARGET_PARTS)
            combined_mask = tf.maximum(target_mask, look_ahead_mask)

            # add <EOS>
            EOS_target_indices = tf.reduce_sum(tf.cast(tf.not_equal(output_target_ids, 0), tf.int32), axis=-1)
            EOS_batch_indices = tf.range(batch_size)
            EOS_indices = tf.transpose([EOS_batch_indices, EOS_target_indices])
            inrange_mask = tf.less(EOS_target_indices, self.config.MAX_TARGET_PARTS)
            EOS_indices = tf.cast(tf.boolean_mask(EOS_indices, inrange_mask), tf.int64)
            output_target_ids = tf.add(
                output_target_ids,
                tf.sparse.to_dense(
                    tf.SparseTensor(
                        EOS_indices,
                        tf.fill(tf.shape(EOS_indices)[:1], self.queue_thread.EOS_id),
                        (batch_size, self.config.MAX_TARGET_PARTS)
                    )
                )
            )

            target_emb = tf.nn.embedding_lookup(params=target_words_vocab, ids=input_target_ids)
            logits, _ = self.decoder(target_emb, encoded, True, combined_mask, input_mask, self.target_vocab_size)

            step = tf.Variable(0, trainable=False)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_target_ids, logits=logits)
            target_words_nonzero = tf.cast(tf.not_equal(output_target_ids, 0), tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            if self.config.USE_MOMENTUM:
                learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
                                                           self.num_training_examples,
                                                           0.95, staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
                train_op = optimizer.minimize(loss, global_step=step)
            else:
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.saver = tf.train.Saver(max_to_keep=10)

        return train_op, loss

    def prepare_node_embeddings(self, nodes_vocab, node_ids, subtoken_vocab, nodes_subtokens_ids):
        nodes_mask = tf.cast(tf.not_equal(nodes_subtokens_ids, 0), tf.float32)[:, :, :, tf.newaxis]
        nodes_types_emb = tf.nn.embedding_lookup(params=nodes_vocab, ids=node_ids)
        nodes_subtoken_emb = tf.nn.embedding_lookup(params=subtoken_vocab, ids=nodes_subtokens_ids)
        nodes_emb = tf.concat([nodes_types_emb, tf.reduce_sum(nodes_subtoken_emb * nodes_mask, axis=-2)], axis=-1)
        return nodes_emb

    def build_test_graph(self, input_tensors):
        node_ids = input_tensors[reader.NODE_IDS_KEY]
        nodes_subtokens_ids = input_tensors[reader.SUBTOKEN_IDS_KEY]
        incidence_matrix = input_tensors[reader.INCIDENCE_MATRIX_KEY]

        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32, trainable=False)
            nodes_vocab = tf.get_variable('NODES_VOCAB',
                                          shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            batch_size = tf.shape(node_ids)[0]
            input_mask = create_padding_mask(node_ids)
            nodes_emb = self.prepare_node_embeddings(nodes_vocab, node_ids, subtoken_vocab, nodes_subtokens_ids)
            incidence_matrix = incidence_matrix[:, tf.newaxis, :, :]
            encoded = self.encoder(nodes_emb, False, input_mask, incidence_matrix)

            input_target_ids = tf.fill((batch_size, 1), self.queue_thread.SOS_id)

            for i in range(self.config.MAX_TARGET_PARTS):
                input_target_ids_padded = reader.pad_up_to(input_target_ids, (batch_size, self.config.MAX_TARGET_PARTS), 0)
                target_mask = create_padding_mask(input_target_ids_padded)
                look_ahead_mask = create_look_ahead_mask(tf.shape(input_target_ids_padded)[1])
                combined_mask = tf.maximum(target_mask, look_ahead_mask)

                target_emb = tf.nn.embedding_lookup(params=target_words_vocab, ids=input_target_ids_padded)
                logits, _ = self.decoder(target_emb, encoded, True, combined_mask, input_mask, self.target_vocab_size)
                predictions = logits[:, -1:, :]
                predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
                input_target_ids = tf.concat([input_target_ids, predicted_id], axis=-1)
            predicted_indices = input_target_ids[:, 1:]

        return predicted_indices

    def predict(self, predict_data_lines):
        if self.predict_queue is None:
            self.predict_queue = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                               node_to_index=self.node_to_index,
                                               target_to_index=self.target_to_index,
                                               config=self.config, is_evaluating=True)
            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_queue.process_from_placeholder(self.predict_placeholder)
            reader_output = {key: tf.expand_dims(tensor, 0) for key, tensor in reader_output.items()}
            self.predict_top_indices_op, self.predict_top_scores_op, _, self.attention_weights_op = \
                self.build_test_graph(reader_output)
            self.predict_source_string = reader_output[reader.PATH_SOURCE_STRINGS_KEY]
            self.predict_path_string = reader_output[reader.PATH_STRINGS_KEY]
            self.predict_path_target_string = reader_output[reader.PATH_TARGET_STRINGS_KEY]
            self.predict_target_strings_op = reader_output[reader.TARGET_STRING_KEY]

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        results = []
        for line in predict_data_lines:
            predicted_indices, top_scores, true_target_strings, attention_weights, path_source_string, path_strings, path_target_string = self.sess.run(
                [self.predict_top_indices_op, self.predict_top_scores_op, self.predict_target_strings_op,
                 self.attention_weights_op,
                 self.predict_source_string, self.predict_path_string, self.predict_path_target_string],
                feed_dict={self.predict_placeholder: line})

            top_scores = np.squeeze(top_scores, axis=0)
            path_source_string = path_source_string.reshape((-1))
            path_strings = path_strings.reshape((-1))
            path_target_string = path_target_string.reshape((-1))
            predicted_indices = np.squeeze(predicted_indices, axis=0)
            true_target_strings = Common.binary_to_string(true_target_strings[0])

            if self.config.BEAM_WIDTH > 0:
                predicted_strings = [[self.index_to_target[sugg] for sugg in timestep]
                                     for timestep in predicted_indices]  # (target_length, top-k)  
                predicted_strings = list(map(list, zip(*predicted_strings)))  # (top-k, target_length)
                top_scores = [np.exp(np.sum(s)) for s in zip(*top_scores)]
            else:
                predicted_strings = [self.index_to_target[idx]
                                     for idx in predicted_indices]  # (batch, target_length)  

            attention_per_path = None
            if self.config.BEAM_WIDTH == 0:
                attention_per_path = self.get_attention_per_path(path_source_string, path_strings, path_target_string,
                                                                 attention_weights)

            results.append((true_target_strings, predicted_strings, top_scores, attention_per_path))
        return results

    @staticmethod
    def get_attention_per_path(source_strings, path_strings, target_strings, attention_weights):
        # attention_weights:  (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for source, path, target, weight in zip(source_strings, path_strings, target_strings, time_step):
                string_triplet = (
                    Common.binary_to_string(source), Common.binary_to_string(path), Common.binary_to_string(target))
                attention_per_context[string_triplet] = weight
            results.append(attention_per_context)
        return results

    def save_model(self, sess, path):
        save_target = path + '_iter%d' % self.epochs_trained
        dirname = os.path.dirname(save_target)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(sess, save_target)

        dictionaries_path = save_target + '.dict'
        with open(dictionaries_path, 'wb') as file:
            pickle.dump(self.subtoken_to_index, file)
            pickle.dump(self.index_to_subtoken, file)
            pickle.dump(self.subtoken_vocab_size, file)

            pickle.dump(self.target_to_index, file)
            pickle.dump(self.index_to_target, file)
            pickle.dump(self.target_vocab_size, file)

            pickle.dump(self.node_to_index, file)
            pickle.dump(self.index_to_node, file)
            pickle.dump(self.nodes_vocab_size, file)

            pickle.dump(self.num_training_examples, file)
            pickle.dump(self.epochs_trained, file)
            pickle.dump(self.config, file)
        print('Saved after %d epochs in: %s' % (self.epochs_trained, save_target))

    def load_model(self, sess):
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done loading model')
        with open(self.config.LOAD_PATH + '.dict', 'rb') as file:
            if self.subtoken_to_index is not None:
                return
            print('Loading dictionaries from: ' + self.config.LOAD_PATH)
            self.subtoken_to_index = pickle.load(file)
            self.index_to_subtoken = pickle.load(file)
            self.subtoken_vocab_size = pickle.load(file)

            self.target_to_index = pickle.load(file)
            self.index_to_target = pickle.load(file)
            self.target_vocab_size = pickle.load(file)

            self.node_to_index = pickle.load(file)
            self.index_to_node = pickle.load(file)
            self.nodes_vocab_size = pickle.load(file)

            self.num_training_examples = pickle.load(file)
            self.epochs_trained = pickle.load(file)
            saved_config = pickle.load(file)
            self.config.take_model_hyperparams_from(saved_config)
            print('Done loading dictionaries')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
