# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on DuReader."""

from __future__ import absolute_import, division, print_function

import collections
import json
import math
import os
import pdb
import pickle
import random

import horovod.tensorflow as hvd
import numpy
import six
import tensorflow as tf
from tqdm import tqdm

import modeling
import optimization
import tokenization

# 这里为了避免打印重复的日志信息
tf.get_logger().propagate = False

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "data_dir", None,
    "The output directory where the data will be loaded and written.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "predict_dir", None,
    "The output directory where the predictions will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "eval_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("num_accumulation_steps", 1, "how many steps to do gradients accumulation")

flags.DEFINE_bool("horovod", False, "Whether to use horovod for distribute training.")

flags.DEFINE_bool("amp", False, "Whether to use auto mix-precision for training.")

flags.DEFINE_bool("xla", False, "Whether to use xla for training.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer("rand_seed", 12345, "set random seed")

# set random seed (i don't know whether it works or not)
numpy.random.seed(int(FLAGS.rand_seed))
tf.set_random_seed(int(FLAGS.rand_seed))

#
class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               ans_doc=None,
               fake_docs=[]):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.ans_doc = ans_doc
    self.fake_docs = fake_docs

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens[0]))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.ans_doc:
      s += ', ans_doc: %d' % (self.ans_doc)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               input_span_mask,
               start_position=None,
               end_position=None,
               ans_doc=0):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_span_mask = input_span_mask
    self.start_position = start_position
    self.end_position = end_position
    self.ans_doc = ans_doc

#
def customize_tokenizer(text, do_lower_case=False):
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
  temp_x = ""
  text = tokenization.convert_to_unicode(text)
  for c in text:
    if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
      temp_x += " " + c + " "
    else:
      temp_x += c
  if do_lower_case:
    temp_x = temp_x.lower()
  return temp_x.split()

#
class ChineseFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False):
    self.vocab = tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
    self.do_lower_case = do_lower_case
  def tokenize(self, text):
    split_tokens = []
    for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)

#
def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  tf.logging.info('read squad data')
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  tf.logging.info('preprocess examples')
  examples = []
  failed_cnt = 0
  for entry_idx,entry in enumerate(input_data):
    if (entry_idx + 1) % 5000 == 0:
      tf.logging.info(f'Processes {entry_idx + 1} data')
    for paragraph in entry["paragraphs"]:
      current_doc_tokens = []
      current_char_to_word_offset = []
      current_paragraph_text = []
      failed = False
      for paragraph_text in paragraph['context']:
        raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=FLAGS.do_lower_case)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        k = 0
        temp_word = ""
        for c in paragraph_text:
          if tokenization._is_whitespace(c):
            char_to_word_offset.append(k-1)
            continue
          else:
            temp_word += c
            char_to_word_offset.append(k)
          if FLAGS.do_lower_case:
            temp_word = temp_word.lower()
          if temp_word == raw_doc_tokens[k]:
            doc_tokens.append(temp_word)
            temp_word = ""
            k += 1

        try:
          assert k==len(raw_doc_tokens)
        except:
          tf.logging.warning('Error with {}'.format(paragraph['id']))
          failed = True
          break

        current_paragraph_text.append(paragraph_text)
        current_doc_tokens.append(raw_doc_tokens)
        current_char_to_word_offset.append(char_to_word_offset)

      if failed:
        failed_cnt += 1
        continue

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        ans_doc = qa['ans_doc']
        # Only valid in predict
        fake_docs = qa.get('fake_docs',[])

        start_position = None
        end_position = None
        orig_answer_text = None

        if is_training:
          paragraph_text = current_paragraph_text[ans_doc]
          char_to_word_offset = current_char_to_word_offset[ans_doc]
          doc_tokens = current_doc_tokens[ans_doc]
          answer = qa["answers"][0]
          orig_answer_text = answer["text"]

          if orig_answer_text not in paragraph_text:
            tf.logging.warning("Could not find answer")
          else:
            answer_offset = paragraph_text.index(orig_answer_text)
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = "".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = "".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if FLAGS.do_lower_case:
                cleaned_answer_text = cleaned_answer_text.lower()
            if actual_text.find(cleaned_answer_text) == -1:
              # you should never reach here !!! 
              pdb.set_trace()
              tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
              continue

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=current_doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            ans_doc=ans_doc,
            fake_docs=fake_docs)
        examples.append(example)
  tf.logging.info("**********read_squad_examples complete!**********")
  tf.logging.info(f'failed examples size {failed_cnt}')
  
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  tf.logging.info(f'start to convert {len(examples)} examples to features')
  for (example_index, example) in enumerate(examples):
    if (example_index + 1) % 5000 == 0:
      tf.logging.info(f'converted {example_index + 1} examples to features')
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    def create_features(doc_tokens, is_training=True):
      tok_to_orig_index = []
      orig_to_tok_index = []
      all_doc_tokens = []
      for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
          tok_to_orig_index.append(i)
          all_doc_tokens.append(sub_token)

      tok_start_position = None
      tok_end_position = None
      if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(doc_tokens) - 1:
          tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
          tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

      # The -3 accounts for [CLS], [SEP] and [SEP]
      max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

      # We can have documents that are longer than the maximum sequence length.
      # To deal with this we do a sliding window approach, where we take chunks
      # of the up to our max length with a stride of `doc_stride`.
      _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
          "DocSpan", ["start", "length"])
      doc_spans = []
      start_offset = 0
      while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
          length = max_tokens_for_doc
          
        doc_span = _DocSpan(start=start_offset, length=length)
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if is_training:
          if not (tok_start_position >= doc_start and
                  tok_end_position <= doc_end):
            out_of_span = True
        
        if out_of_span:
          if start_offset + length == len(all_doc_tokens):
            break
          start_offset += min(length, doc_stride)
        else:
          # 训练时我们不想处理复杂的doc_span问题，只取包含正确答案的span就好了
          # 预测时候为了方便我们也就只取第一个span
          doc_spans.append(doc_span)
          break

      if not doc_spans:
        return None

      for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        input_span_mask = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        input_span_mask.append(1)
        for token in query_tokens:
          tokens.append(token)
          segment_ids.append(0)
          input_span_mask.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_span_mask.append(0)

        for i in range(doc_span.length):
          split_token_index = doc_span.start + i
          token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

          is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                split_token_index)
          token_is_max_context[len(tokens)] = is_max_context
          tokens.append(all_doc_tokens[split_token_index])
          segment_ids.append(1)
          input_span_mask.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_span_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)
          input_span_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_span_mask) == max_seq_length

        start_position = None
        end_position = None
        if is_training:
          # For training, if our document chunk does not contain an annotation
          # we throw it out, since there is nothing to predict.
          doc_start = doc_span.start
          doc_end = doc_span.start + doc_span.length - 1
          out_of_span = False
          if not (tok_start_position >= doc_start and
                  tok_end_position <= doc_end):
            out_of_span = True
          if out_of_span:
            tf.logging.warning('out_of_span %s' %(example_index))
            return None
            start_position = 0
            end_position = 0
          else:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

        if example_index < 1:
          tf.logging.info("*** Example ***")
          tf.logging.info("unique_id: %s" % (unique_id))
          tf.logging.info("example_index: %s" % (example_index))
          tf.logging.info("doc_span_index: %s" % (doc_span_index))
          tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.logging.info("token_to_orig_map: %s" % " ".join(
              ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
          tf.logging.info("token_is_max_context: %s" % " ".join([
              "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
          ]))
          tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.logging.info(
              "input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.logging.info(
              "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
          tf.logging.info(
            "input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
          if is_training:
            answer_text = " ".join(tokens[start_position:(end_position + 1)])
            tf.logging.info("start_position: %d" % (start_position))
            tf.logging.info("end_position: %d" % (end_position))
            tf.logging.info(
                "answer: %s" % (tokenization.printable_text(answer_text)))


        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_span_mask=input_span_mask,
            start_position=start_position,
            end_position=end_position)
        return feature

    all_features = []
    success = True
    for i,doc_tokens in enumerate(example.doc_tokens):
      if i == example.ans_doc:
        features = create_features(doc_tokens,is_training)
      else:
        features = create_features(doc_tokens, False)

      if is_training and features is None:
        success = False
        break 

      all_features.append(features)

    if not success:
      continue

    final_features = InputFeatures(
      unique_id=unique_id,
      example_index=example_index,
      doc_span_index=0, # 我们只用一个doc_span
      tokens=[],
      token_to_orig_map=[],
      token_is_max_context=[],
      input_ids=[],
      input_mask=[],
      segment_ids=[],
      input_span_mask=[],
      start_position=None,
      end_position=None)

    start = 0
    end = 0
    for i,features in enumerate(all_features):
      if is_training:
        if i == example.ans_doc:
          final_features.start_position = start + features.start_position
          final_features.end_position = end + features.end_position
          final_features.ans_doc = i
        else:
          start += len(features.input_ids)
          end += len(features.input_ids)
      
      final_features.tokens.append(features.tokens)
      final_features.token_to_orig_map.append(features.token_to_orig_map)
      final_features.token_is_max_context.append(features.token_is_max_context)
      final_features.input_ids.extend(features.input_ids)
      final_features.input_mask.extend(features.input_mask)
      final_features.segment_ids.extend(features.segment_ids)
      final_features.input_span_mask.extend(features.input_span_mask)

    if is_training:
      try:
        assert final_features.input_span_mask[final_features.start_position] == 1
        assert final_features.input_span_mask[final_features.end_position] == 1
      except:
        # you must not reach here !!!
        pdb.set_trace()
    # Run callback
    output_fn(final_features)

    unique_id += 1  


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


#
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, input_span_mask,
                 use_one_hot_embeddings=False):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)


  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # apply output mask
  adder           = (1.0 - tf.cast(input_span_mask, tf.float32)) * -100000.0
  start_logits   += adder
  end_logits     += adder

  output_layer = model.get_pooled_output()
  if is_training:
    # I.e., 0.1 dropout
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
  
  logits = tf.layers.dense(output_layer,1,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='transform')


  return (start_logits, end_logits, logits)


#
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, hvd=None, amp=False, num_docs=5):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    # Batch * num_sample * size
    def reshape_features(feature, batch_size):
      return tf.reshape(feature,[batch_size*num_docs,-1])

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    input_span_mask = features["input_span_mask"]

    batch_size = tf.shape(input_ids)[0]
    input_ids = reshape_features(input_ids,batch_size)
    input_mask = reshape_features(input_mask,batch_size)
    segment_ids = reshape_features(segment_ids,batch_size)
    input_span_mask = reshape_features(input_span_mask,batch_size)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits, logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        input_span_mask=input_span_mask)

    start_logits = tf.reshape(start_logits,[batch_size,-1])
    end_logits = tf.reshape(end_logits,[batch_size,-1])
    logits = tf.reshape(logits,[batch_size,num_docs])

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        on_hot_pos    = tf.one_hot(positions, depth=seq_length*num_docs, dtype=tf.float32)
        log_probs     = tf.nn.log_softmax(logits, axis=-1)
        loss          = -tf.reduce_mean(tf.reduce_sum(on_hot_pos * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions   = features["end_positions"]

      start_loss  = compute_loss(start_logits, start_positions)
      end_loss    = compute_loss(end_logits, end_positions)
      mrc_loss  = (start_loss + end_loss)

      
      probabilities = tf.nn.softmax(logits, axis=-1)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      labels = features['ans_doc']

      one_hot_labels = tf.one_hot(labels, depth=num_docs, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      rank_loss = tf.reduce_mean(per_example_loss)

      total_loss = rank_loss + mrc_loss

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd, amp, FLAGS.num_accumulation_steps)
      
      # Some metrics to monitor
      accuracy = tf.metrics.accuracy(labels,tf.arg_max(log_probs,dimension=-1))
      start_accuracy = tf.metrics.accuracy(start_positions,tf.arg_max(start_logits,dimension=-1))
      end_accuracy = tf.metrics.accuracy(end_positions,tf.arg_max(end_logits,dimension=-1))

      tensor_to_log = {
        'mrc_loss': mrc_loss, 
        "rank_loss":rank_loss,
        'rank_acc':accuracy[1],
        "start_acc":start_accuracy[1],
        "end_acc":end_accuracy[1]
        }

      tf.summary.scalar('mrc_loss',mrc_loss)
      tf.summary.scalar('rank_loss',rank_loss)
      # 我们在前面已经做了update的操作，这边只需要拿到结果即可
      tf.summary.scalar('rank_acc',accuracy[0])
      tf.summary.scalar('start_acc',start_accuracy[0])
      tf.summary.scalar('end_acc',end_accuracy[0])

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[tf.train.LoggingTensorHook(tensor_to_log, every_n_iter=50)])
    elif mode == tf.estimator.ModeKeys.PREDICT:
      start_logits = tf.reshape(start_logits,[batch_size,num_docs,-1])
      end_logits = tf.reshape(end_logits,[batch_size,num_docs,-1])
      start_logits = tf.nn.log_softmax(start_logits, axis=-1)
      end_logits = tf.nn.log_softmax(end_logits, axis=-1)

      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "doc_logits": logits
      }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, batch_size,seq_length, is_training, drop_remainder, hvd=None,num_docs=5):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length*num_docs], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length*num_docs], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length*num_docs], tf.int64),
      "input_span_mask": tf.FixedLenFeature([seq_length*num_docs], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features['ans_doc'] = tf.FixedLenFeature([], tf.int64)


  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.TFRecordDataset(input_file, num_parallel_reads=4)
      if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
      d = d.apply(tf.data.experimental.ignore_errors())
      d = d.shuffle(buffer_size=100)
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_file)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "doc_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "doc_index", "doc_prob", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]
    prelim_predictions = []
    
    for (feature_index, feature) in enumerate(features):  # multi-trunk
      result = unique_id_to_result[feature.unique_id]

      fake_docs = example.fake_docs

      doc_logits = [logits-1e20 if i in fake_docs else logits for i,logits in enumerate(result.doc_logits)]
      doc_probs = _compute_softmax(doc_logits)
      doc_log_probs = [math.log(p) for p in doc_probs]

      for i,(start_logits,end_logits) in enumerate(zip(result.start_logits, result.end_logits)):
        start_indexes = _get_best_indexes(start_logits, n_best_size)
        end_indexes = _get_best_indexes(end_logits, n_best_size)

        for start_index in start_indexes:
          for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens[i]):
              continue
            if end_index >= len(feature.tokens[i]):
              continue
            if start_index not in feature.token_to_orig_map[i]:
              continue
            if end_index not in feature.token_to_orig_map[i]:
              continue
            if not feature.token_is_max_context[i].get(start_index, False):
              continue
            if end_index < start_index:
              continue
            length = end_index - start_index + 1
            if length > max_answer_length:
              continue
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    doc_index=i,
                    doc_prob=doc_probs[i],
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=doc_log_probs[i] + start_logits[start_index],
                    end_logit=doc_log_probs[i] + end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index", "doc_index", "doc_prob"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      doc_index = pred.doc_index
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[doc_index][pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[doc_index][pred.start_index]
        orig_doc_end = feature.token_to_orig_map[doc_index][pred.end_index]
        orig_tokens = example.doc_tokens[doc_index][orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        final_text = final_text.replace(' ','')
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              start_index=pred.start_index,
              end_index=pred.end_index,
              doc_index=pred.doc_index,
              doc_prob=pred.doc_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0, doc_index=0, doc_prob=1.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_index"] = entry.start_index
      output["end_index"] = entry.end_index
      output['doc_index'] = entry.doc_index
      output['doc_prob'] = entry.doc_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    all_predictions[example.qas_id] = best_non_null_entry.text
    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=2, ensure_ascii=False) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=2, ensure_ascii=False) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    if score == 0.0:
      score = 1e-20
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["input_span_mask"] = create_int_feature(feature.input_span_mask)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["ans_doc"] = create_int_feature([feature.ans_doc])


    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")
  if FLAGS.do_eval:
    if not FLAGS.eval_file:
      raise ValueError(
          "If `do_eval` is True, then `eval_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  if FLAGS.horovod:
    hvd.init()

  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = ChineseFullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  session_config = tf.ConfigProto()
  master_process = True
  hvd_rank = 0
  training_hooks = []
  global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
  if FLAGS.horovod:
    hvd_rank = hvd.rank()
    master_process = (hvd_rank == 0)
    global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps * hvd.size()
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    if hvd.size() > 1:
      training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if FLAGS.xla:
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  # 如果这个设置没有生效，那么还可以配置环境变量TF_FORCE_GPU_ALLOW_GROWTH=true
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  # 如果只有一个gpu那么就不使用分布式
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir if master_process else None,
      keep_checkpoint_max=1,
      log_step_count_steps=50,
      session_config=session_config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    example_path = os.path.join(FLAGS.data_dir, "train.examples")
    if os.path.exists(example_path):
      tf.logging.info('Read data from pre saved file')
      train_examples = pickle.load(open(example_path,'rb'))
    else:
      train_examples = read_squad_examples(input_file=FLAGS.train_file, is_training=True)
      random.shuffle(train_examples)
      pickle.dump(train_examples, open(os.path.join(FLAGS.data_dir, "train.examples"),'wb'))

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    start_index = 0
    end_index = len(train_examples)
    record_paths = [os.path.join(FLAGS.data_dir, "train.tf_record")]
    record_path = record_paths[0]
    if FLAGS.horovod:
      record_paths = [os.path.join(FLAGS.data_dir, f"train.tf_record{i}") for i in range(hvd.size())]
      num_examples_per_rank = len(train_examples) // hvd.size()
      remainder = len(train_examples) % hvd.size()
      if hvd_rank < remainder:
        start_index = hvd_rank * (num_examples_per_rank+1)
        end_index = start_index + num_examples_per_rank + 1
      else:
        start_index = hvd_rank * num_examples_per_rank + remainder
        end_index = start_index + (num_examples_per_rank)
      
      record_path = record_paths[hvd_rank]

    if os.path.exists(record_path):
      num_features = 0
      for record in tf.python_io.tf_record_iterator(record_path):
        num_features += 1
    else:
      train_writer = FeatureWriter(
          filename=record_path,
          is_training=True)
      convert_examples_to_features(
          examples=train_examples[start_index:end_index],
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=True,
          output_fn=train_writer.process_feature)
      train_writer.close()
      num_features = train_writer.num_features

    train_examples_len = len(train_examples)
    del train_examples

    num_train_steps = int(train_examples_len / global_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", end_index - start_index)
    tf.logging.info("  Num split examples = %d", num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      hvd=hvd if FLAGS.horovod else None,
      amp=FLAGS.amp)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  # do training
  if FLAGS.do_train:
    train_input_fn = input_fn_builder(
        input_file=record_paths,
        batch_size=FLAGS.train_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        hvd=hvd if FLAGS.horovod else None)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  # do predictions
  if FLAGS.do_predict:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)
    predict_filename = os.path.join(FLAGS.predict_dir, "predict.tf_record")
    eval_writer = FeatureWriter(
        filename=predict_filename,
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
        input_file=predict_filename,
        batch_size=FLAGS.predict_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 100 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [[float(x) for x in y.flat] for y in result["start_logits"]]
      end_logits = [[float(x) for x in y.flat] for y in result["end_logits"]]
      rank_logits = [float(x) for x in result['doc_logits']]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits,
              doc_logits=rank_logits))

    output_json_name = "predict_predictions.json"
    output_nbest_name = "predict_nbest_predictions.json"

    output_prediction_file = os.path.join(FLAGS.predict_dir, output_json_name)
    output_nbest_file = os.path.join(FLAGS.predict_dir, output_nbest_name)

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file)

if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("data_dir")
  tf.app.run()



