require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const _ = require('lodash');

let {features, labels, testFeatures, testLabels} = loadCSV('trex.csv', 
        {
            shuffle: true,
            splitTest: 0,
            dataColumns: ['xPos', 'yPos', 'width', 'speed'],
            labelColumns: ['jump'],
            converters: {
                   }
});


const {mean, variance} = tf.moments(features,0);

standarize = (inpt) => {
    inpt = tf.tensor(inpt);
   
    return inpt.sub(mean).div(variance.pow(0.5)); 
  };

//   console.log("feat",features)
//   console.log("lab",labels)
// console.log("mean",mean.print())
// console.log("variance",variance.print())
// console.log("stdfeat",standarize(features).print())

const xs = standarize(features);
const ys = tf.tensor(labels);
// const xt = standarize(testFeatures);
// const yt = tf.tensor(testLabels);


// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 16, inputShape: [4], activation: 'relu'}));
model.add(tf.layers.dense({units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
model.compile({optimizer: 'adam',loss: tf.losses.logLoss});

// Train model with fit().
train = async () => {
      for (let i = 1; i < 20; ++i) {
       const z = await model.fit(xs, ys, {batchSize: 10, epochs: 10});
       console.log("Loss after Epoch " + i + " : " + z.history.loss[0]);
      }
  };

saveResult = async () =>{
//await model.save('localstorage://my-model-trex')
await model.save('file://./model')
};

test = async (textX,testY) => {

    const predictictions =  model.predict(xt).round().squeeze();
    console.log("predictictions:",predictictions.print());

    console.log("testLabels:",yt.squeeze().print());

    const incorrect = predictictions
        .notEqual(yt.squeeze())
        .sum()
        .get();

        const accuracy =  (predictictions.shape[0] - incorrect) / predictictions.shape[0]; 

        console.log("Accuracy on test set:", accuracy);
};


const xSinglePred = standarize(([[82, 105, 34,	6.08100000000023
]]));

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.argMax(1).get(0));
};

exec = async () => {
    await train();
    await saveResult();
    //await test();
    //SinglePredictiction();
  };

exec();

