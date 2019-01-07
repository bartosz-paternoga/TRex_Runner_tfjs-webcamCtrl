import * as tf from '@tensorflow/tfjs';
import Webcam from './webcam';

const webCtrl = () => {


  let mobilenet;
  let model;
  //let predd = [];



  const NUM_CLASSES = 3;
  const webcam = new Webcam(document.getElementById('webcam'));


  // Loads mobilenet and returns a model that returns the internal activation
  // we'll use as input to our classifier model.
  async function loadMobilenet() {

    model = await tf.loadModel('/model2/model.json');
    const mobilenet = await tf.loadModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  }


  async function init() {

    await webcam.setup();
    mobilenet = await loadMobilenet();

    // Warm up the model. This uploads weights to the GPU and compiles the WebGL
    // programs so the first time we collect data from the webcam it will be
    // quick.

    tf.tidy(() => mobilenet.predict(webcam.capture()));
      
    predict();

  }

  // Initialize the application.
  init();

  
  async function predict() {


  while (true) {

    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      //return predictions.as1D().argMax();
      return predictions.as1D();
    });

    const classId = (await (predictedClass.argMax()).data())[0];

    const prob = (await (predictedClass.dataSync())[classId]);

    console.log("classId:", classId);
    console.log("prob:", prob);

if( classId === 2 ) {

  simulateKey(38); // arrow up
}


    function simulateKey (keyCode, type, modifiers) {
      var evtName = (typeof(type) === "string") ? "key" + type : "keydown";	
      var modifier = (typeof(modifiers) === "object") ? modifier : {};

      var event = document.createEvent("HTMLEvents");
      event.initEvent(evtName, true, false);
      event.keyCode = keyCode;
      
      for (var i in modifiers) {
        event[i] = modifiers[i];
      }

      document.dispatchEvent(event);
    }

     const elem = document.getElementById("Div1");
      
      if (elem !== null){
      elem.parentNode.removeChild(elem);
    }

      const div = document.createElement('div');
      div.setAttribute("id", "Div1");
      document.body.appendChild(div);
      div.style.marginBottom = '10px';
      // Create info text


      const infoText = document.createElement('span')
      infoText.innerText = (`THE PREDICTED CLASS: ${classId} with prob. ${(prob*100).toFixed(0)}%`);
      div.appendChild(infoText);



    await tf.nextFrame();
  

    }

  }


}

export default webCtrl;