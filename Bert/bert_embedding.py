import json
import modeling
import tokenization
import tensorflow as tf
import collections
from NSM_bert_dataset import NSMDataset

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", './Data/ratings_train.txt', "")

flags.DEFINE_string("output_file", 'output.jsonl', "")

flags.DEFINE_string("layers", "-1", "")

flags.DEFINE_string(
    "bert_config_file", '../multi_cased_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 20,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", '../multi_cased_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", '../multi_cased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

    def __init__(self, unique_id, text_a):
        self.unique_id = unique_id
        self.text_a = text_a


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)

        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold

        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {"unique_id": unique_ids, }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(InputFeatures(
                                        unique_id=example.unique_id,
                                        tokens=tokens,
                                        input_ids=input_ids,
                                        input_mask=input_mask,
                                        input_type_ids=input_type_ids))

    return features


def read_examples(reviews):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0

    for line in reviews:
        line = tokenization.convert_to_unicode(line)
        line = line.strip()
        text_a = line
        examples.append(InputExample(unique_id=unique_id, text_a=text_a))
        unique_id += 1
    return examples


def bert_preprocess(reviews):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    examples = read_examples(reviews)

    features = convert_examples_to_features(examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)
    all_features = []

    for result in estimator.predict(input_fn, yield_single_examples=True):

        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]

        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["values"] = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
                all_layers.append(layers)
            features = collections.OrderedDict()
            features["token"] = token
            features["layers"] = all_layers
            all_features.append(features)

    return all_features


if __name__ == '__main__':

    print('pre-loading data')

    a = NSMDataset('./Data/ratings_test.txt', 20)
    reviews = a.reviews

    all_features = bert_preprocess(reviews)
    print()