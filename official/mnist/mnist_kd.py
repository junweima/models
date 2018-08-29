""" knowledgify extension to mnist """

__author__ = 'Jeremy Ma'

import tensorflow as tf
from absl import flags
from official.mnist import dataset
import pdb

flags.DEFINE_string('mnist_data_dir', './mnist_data', 'mnist data directory')
flags.DEFINE_string('mnist_model_dir', './mnist_model', 'mnist model directory')
flags.DEFINE_integer('batch_size', 100, None)
flags.DEFINE_integer('train_epochs', 3, None)
flags.DEFINE_float('learning_rate', 1e-4, None)
flags.DEFINE_string('data_format', 'channels_last', None)

FLAGS = flags.FLAGS

def train_input_fn():
    ds = dataset.train(FLAGS.mnist_data_dir)
    ds = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size)
    ds = ds.repeat(1) # logging every epoch
    return ds


def eval_input_fn():
    return dataset.test(FLAGS.mnist_data_dir).batch(
        FLAGS.batch_size).make_one_shot_iterator().get_next()


def train_model(estimator):
    # train model
    train_hooks = tf.train.LoggingTensorHook([], at_end=True)
    for _ in range(FLAGS.train_epochs):
        estimator.train(
            input_fn=train_input_fn,
            hooks=[train_hooks]
        )
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

    # TODO: export serving model
    # if FLAGS.mnist_model_dir is not None:


def train_distill():
    pass


def create_teacher_model(data_format):
    """Create teacher model

    Args:
        data_format: 'channels_first' (faster on GPU) or 'channels_last' (faster on CPU)

    Returns:
        tf.keras.Model
    """
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        input_shape = [28, 28, 1]

    l = tf.keras.layers
    max_pool = l.MaxPool2D(
        (2, 2), (2, 2), padding='same', data_format=data_format
    )

    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28*28,)
            ),
            l.Conv2D(
                32,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu
            ),
            max_pool,
            l.Conv2D(
                64,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu
            ),
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10)
        ]
    )


def teacher_model_fn(features, labels, mode, params):
    """ model_fn for teacher """
    model = create_teacher_model(params['data_format'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        logits = model(features, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            }
        )
    

def create_student_model(data_format):
    """Create student model

    Args:
        data_format: 'channels_first' (faster on GPU) or 'channels_last' (faster on CPU)

    Returns:
        tf.keras.Model
    """

    l = tf.keras.layers
    return tf.keras.Sequential(
        [
            l.Dense(10)
        ]
    )


def student_model_fn(features, labels, mode, params):
    """ model_fn for student """
    model = create_student_model(params['data_format'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        logits = model(features, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            }
        )



def main(_):
    """ create teacher and student models and then run distillation """
    teacher_classifier = tf.estimator.Estimator(
        model_fn=teacher_model_fn,
        model_dir=FLAGS.mnist_model_dir,
        params={
            'data_format': FLAGS.data_format
        }
    )

    student_classifier = tf.estimator.Estimator(
        model_fn=teacher_model_fn,
        model_dir=FLAGS.mnist_model_dir,
        params={
            'data_format': FLAGS.data_format
        }
    )

    train_model(student_classifier)
    train_distill()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)








