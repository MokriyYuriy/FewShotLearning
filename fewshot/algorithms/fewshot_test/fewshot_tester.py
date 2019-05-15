import os
import tqdm
import datetime

import numpy as np
import tensorflow as tf


def baseline_fewshot_test(model,
                          generator,
                          fewshot_train_args,
                          fewshot_predict_args,
                          n_episodes=10000,
                          model_name='baseline',
                          tensorboard=False,
                          log_dir='../fewshot/logs',
                          period=True):

    if tensorboard:
        log_dir = os.path.join(log_dir,
            '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        os.makedirs(log_dir, exist_ok=True)
        file_writer = tf.summary.FileWriter(log_dir)

    accuracies = []
    tbar = tqdm.tqdm(range(n_episodes), total=n_episodes)
    for episode_index in tbar:
        (support_x, support_y), (query_x, query_y) = generator[episode_index]
        model.fit(support_x, support_y, **fewshot_train_args)

        out = model.predict(query_x, **fewshot_predict_args)

        accuracy = np.mean(np.argmax(out, axis=1) == np.where(query_y == 1)[1])
        accuracies.append(accuracy)

        tbar.set_description(
            "Average acc: {:.2f}%".format(np.mean(accuracies) * 100))

        if tensorboard and (episode_index % period == 0):
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                tf.Summary.Value(tag='average_accuracy', simple_value=np.mean(accuracies))
            ])
            file_writer.add_summary(summary, global_step=episode_index)
            file_writer.flush()

    return accuracies
