import progressbar
import tensorflow as tf

class mnist_application:
    def __init__(self, model, dataset):
        self.__model = model
        self.__dataset = dataset

        self.__classifier = self.__model(self.__dataset.shape, self.__dataset.images_dtype, self.__dataset.labels_dtype)

    def run(self, withTrain=True):
        trainer_step = self.__classifier.trainer_step

        merged = tf.summary.merge_all()
        logdir = "./logdir/{0}/".format(str(self.__model).split("'")[1].split(".")[1])
        train_writer = tf.summary.FileWriter(logdir)
        train_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore Model
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                train_saver.restore(sess, ckpt.model_checkpoint_path)
                print("Previous Session Restored")
            
            if withTrain:
                # Model Training
                print("Training Model...")
                bar = progressbar.ProgressBar()
                batch_gen = self.__dataset.train.get_random_batcher(BATCH_SIZE)
                for i in bar(range(EPOCHS), max_value=EPOCHS):
                    batch = next(batch_gen)
                    _, summary = sess.run([trainer_step, merged], feed_dict=self.__classifier.get_feed_dict(batch[0], batch[1], True))
                    train_writer.add_summary(summary, i)

                    if i % 1000 == 999:
                        train_saver.save(sess, logdir+"model.ckpt", i)

            # Modles Evaluation
            ok = 0
            batch_gen = self.__dataset.eval.get_batcher(BATCH_SIZE)
            for batch in batch_gen:
                ok += self.__classifier.evaluator.eval(feed_dict=self.__classifier.get_feed_dict(batch[0], batch[1], False))
            print("Précision : {0:.2f}%".format(100*ok/self.__dataset.eval.size))

        train_writer.close()
        input("Press <ENTER> to continue")

if __name__ == "__main__":
    from data_import import mnist_dataset
    from model.mlp_model import MLP
    from model.cnn_model import CNN

    #Hyperparameter
    BATCH_SIZE = 60
    EPOCHS = 10000

    #Run app
    mnist = mnist_dataset("./input_data/")
    app = mnist_application(CNN, mnist)
    app.run()
