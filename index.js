"use strict";

const canvas = document.getElementById("xorCanvas");
const ctx = canvas.getContext("2d");
ctx.fillRect(0, 0, canvas.width, canvas.height);

const train_xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const train_ys = tf.tensor2d([0, 1, 1, 0], [4, 1]);

//can try AND and OR gate too, by replacing train_ys(labels) in the model's fit function
const train_ys_and = tf.tensor2d([0, 0, 0, 1], [4, 1]);
const train_ys_or = tf.tensor2d([0, 1, 1, 1], [4, 1]);

//fc hidden layer
const model = tf.sequential();
model.add(
  tf.layers.dense({
    inputShape: [2],
    units: 4,
    activation: "sigmoid",
    useBias: true
  })
);
//fc output layer
model.add(
  tf.layers.dense({
    units: 1,
    activation: "sigmoid"
  })
);

//compile the model!
const LEARNING_RATE = 0.1;
const optimizer = tf.train.adam(LEARNING_RATE);

model.compile({
  optimizer: optimizer,
  loss: tf.losses.meanSquaredError,
  metrics: ["accuracy"]
});

// const BATCH_SIZE = 4;
const EPOCHS = 10;
const TRAIN_BATCHES = 20;

const resolution = 1;
const width = canvas.width;
const height = canvas.height;
const rows = width / resolution;
const cols = height / resolution;
let xs;

//data we'll make predictions on and paint on canvas
async function inputData() {
  let inputs = [];
  for (var i = 0; i < cols; i++) {
    for (var j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }
  xs = tf.tensor2d(inputs);
}

//This is the 'learn' in Machine Learning
async function train() {
  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const history = await model.fit(train_xs, train_ys, {
      shuffle: true,
      epochs: EPOCHS
    });
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];
    console.log("loss " + loss + " accuracy " + accuracy);
  }
}

//Alright, time to check what the machine learned
async function showPredictions() {
  //   testing with one input
  //   var input = tf.tensor2d([[0, 1]]);
  //   var output = model.predict(input);
  //   console.log(output.dataSync());
  await inputData();
  tf.tidy(() => {
    //xs.print();
    let ys = model.predict(xs);
    let y_values = ys.dataSync();

    var imageData = ctx.getImageData(0, 0, width, height);
    var data = imageData.data;

    for (var i = 0; i < data.length; i += 4) {
      var avg = Math.round(y_values[i] * 255);
      data[i] = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      //making it all grayscale
    }
    //I'm not predicting as many data as I require for my canvas, hence all this scaling business
    //Also I feel I'm doing something wrong
    var newCanvas = document.createElement("canvas");
    newCanvas.width = imageData.width;
    newCanvas.height = imageData.height;
    newCanvas.getContext("2d").putImageData(imageData, 0, 0);

    ctx.scale(4, 4);
    ctx.drawImage(newCanvas, 0, 0);
    console.log("canvas painted....");
  });
}
async function XORfunction() {
  await train();
  await showPredictions();
}
XORfunction(); //Let's run this
