import tf from "@tensorflow/tfjs-node";

const model = tf.sequential(); // sequential is a linear stack of layers
// dense unit 1 and input shape 1
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
// compile model
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
// sgd is stochastic gradient descent

// training data
const xs = tf.tensor1d([
  -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
]);
const ys = tf.tensor1d([
  -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0,
]);

// train model
model.fit(xs, ys, { epochs: 500 }).then(() => {
  // predict model
  model.predict(tf.tensor1d([10.0])).print();

  // 신경망이 학습한 것.
  console.log(model.getWeights().map((w) => w.dataSync()));
});

// 학습을 여러번 하거나, 데이터의 양을 늘리면 더 정확한 예측을 할 수 있습니다.
