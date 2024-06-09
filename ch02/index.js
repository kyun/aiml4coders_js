import tf from "@tensorflow/tfjs-node";

// get fashion mnist data
const data = tf.data.csv(
  "https://media.githubusercontent.com/media/fpleoni/fashion_mnist/master/fashion-mnist_train.csv",
  {
    columnConfigs: {
      label: {
        isLabel: true,
      },
    },
  }
);

console.log(data);
const columnNames = await data.columnNames();
const rowCount = await data.rowCount();
console.log(columnNames);
console.log(rowCount);

const model = tf.sequential();
model.add(tf.layers.flatten({ inputShape: [28, 28, 1] })); // 입력을 위한 레이어
model.add(tf.layers.dense({ units: 128, activation: "relu" })); // 은닉층
// 사용할 뉴런의 개수를 지정하는 일정한 규칙은 없다.
// 적절한 뉴런의 개수를 선정해야 한다.
// 뉴런의 개수가 많아지면 학습이 느려지고, 뉴런의 개수가 적으면 학습이 덜 된다.

// 적절한 뉴런의 개수를 선정하는 방법
// 1. 데이터의 양이 많을 때는 뉴런의 개수를 늘려본다.
// 2. 뉴런의 개수를 늘려가면서 학습을 시도해본다.
// 3. 뉴런의 개수를 늘려도 학습이 되지 않는다면, 데이터의 양을 늘려본다.
// 4. 데이터의 양이 많은데도 학습이 되지 않는다면, 뉴런의 개수를 줄여본다.
// 이를 하이퍼파라미터 튜닝이라고 한다.

// ReLU 함수는 입력이 0보다 작으면 0을 반환하고, 0보다 크면 입력을 그대로 반환한다.

model.add(tf.layers.dense({ units: 10, activation: "softmax" })); // 출력층
// softmax 함수는 입력받은 값을 출력으로 0 ~ 1 사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수이다.

// 이 신경망은 28x28 크기의 이미지를 입력으로 받아 10개의 클래스로 분류하는 신경망이다.

model.compile({
  optimizer: "adam",
  loss: "sparseCategoricalCrossentropy",
  metrics: ["accuracy"],
});

// model.fit(inputs, outputs, { epochs: 5 }).then(() => {
//   // model.predict(testImages).print();
//   console.log("Done");
// });
