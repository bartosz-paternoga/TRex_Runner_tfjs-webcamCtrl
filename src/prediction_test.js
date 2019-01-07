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

// const mean = [];
// const variance = [];

// console.log("mean",mean.print())
// console.log("variance",variance.print())

// const mean0 = tf.tensor([197.3648376, 96.8710709, 34.1759377, 7.0023746])
// const variance0 = tf.tensor([23290.1113281, 96.1395264, 227.9279327, 1.0366685])

// console.log("mean0",mean0.print())
// console.log("variance0",variance0.print())

standarize = (inpt) => {
    inpt = tf.tensor(inpt);
   
    return inpt.sub(mean).div(variance.pow(0.5)); 
  };
let model;

modelF = async () =>{
    //await model.save('localstorage://my-model-trex')
    model = await tf.loadModel('file://./model/model.json')
    };


const xSinglePred = standarize(([[108, 105, 34,	6.08100000000023
]]));

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.round().get(0,0));
};

exec = async () => {
    //await train();
    //await saveResult();
    //await test();
    await modelF();
    SinglePredictiction();
  };

exec();

