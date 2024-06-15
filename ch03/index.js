import tf from "@tensorflow/tfjs-node";

const trainingData = tf.data.csv(
  "https://media.githubusercontent.com/media/fpleoni/fashion_mnist/master/fashion-mnist_train.csv",
  {
    columnConfigs: {
      label: {
        isLabel: true,
      },
    },
  }
);

// first 1 row
trainingData.take(1).forEachAsync((e) => console.log(e));
