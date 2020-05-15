import React, { Component } from "react";
import "./Handwriting_Recognization.css";
import { Button } from "../components/Button.jsx";
import * as tf from "@tensorflow/tfjs";
import { MnistData } from "./data";
import { Icon } from "@iconify/react";
import tensorflowIcon from "@iconify/icons-logos/tensorflow";
import { fabric } from "fabric";

let model;
let data;

//traning constants
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
let canvas;

export default class Handwriting_Recognization extends Component {
  constructor() {
    super();
    this.state = {
      trainingComplete: false,
    };
  }

  async componentDidMount() {
    canvas = new fabric.Canvas("canvas", {
      backgroundColor: "rgb(0, 0, 0)",
    });
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.width = 4;
    canvas.freeDrawingBrush.color = "rgb(255, 255, 255)";

    this.createModel();
    await this.load();
    await this.train();
    this.clearCanvas();
  }

  createModel() {
    console.log("Create model");
    model = tf.sequential();
    console.log("Model created");

    console.log("Add layers");
    model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "VarianceScaling",
      })
    );

    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
      })
    );

    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "VarianceScaling",
      })
    );

    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
      })
    );

    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: 10,
        kernelInitializer: "VarianceScaling",
        activation: "softmax",
      })
    );

    console.log("Layers created");

    console.log("Start compiling ...");

    model.compile({
      optimizer: tf.train.sgd(0.15),
      loss: "categoricalCrossentropy",
    });

    console.log("Compiled");
  }

  async load() {
    console.log("Loading MNIST data ...");
    data = new MnistData();
    await data.load();
    console.log("Data loaded successfully");
  }

  async train() {
    console.log("Start training ...");
    for (let i = 0; i < TRAIN_BATCHES; i++) {
      //tidy function ensures no memory leak on the client's side
      const batch = tf.tidy(() => {
        const batch = data.nextTrainBatch(BATCH_SIZE);
        batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
        return batch;
      });

      await model.fit(batch.xs, batch.labels, {
        batchSize: BATCH_SIZE,
        epochs: 1,
      });

      tf.dispose(batch);

      await tf.nextFrame();
    }
    this.setState({ trainingComplete: true });
    console.log("Training complete");
  }

  async predict() {
    let canvasElement = canvas.getElement();

    let tensor = tf.browser
      .fromPixels(canvasElement)
      .resizeNearestNeighbor([28, 28])
      .mean(2)
      .expandDims(2)
      .expandDims()
      .toFloat();

    const out = await model.predict(tensor);

    //await this.draw(img.flatten(), canvasElement);

    console.log(out.dataSync());
  }

  async draw(image, canvas) {
    console.log("here");
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(width, height);
    const data = image.dataSync();

    // set fill color of context
    ctx.fillStyle = "red";

    // create rectangle at a 100,100 point, with 20x20 dimensions
    ctx.fillRect(10, 10, 10, 10);
    await new Promise((r) => setTimeout(r, 1000));

    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  clearCanvas() {
    canvas.clear();
    canvas.setBackgroundColor("rgb(0, 0, 0)");
  }

  render() {
    let loading_predict;
    let drawingCanvas;
    drawingCanvas = (
      <canvas
        //className="drawingCanvas"
        id="canvas"
        width="100"
        height="100"
      ></canvas>
    );
    if (!this.state.trainingComplete) {
      //three dot thing
      loading_predict = (
        <div className="loading">
          <Icon className="icon" icon={tensorflowIcon} />
          <div className="spinner">
            <div className="bounce1"></div>
            <div className="bounce2"></div>
            <div className="bounce3"></div>
          </div>
        </div>
      );
    } else {
      loading_predict = (
        <Button
          onClick={async () => await this.predict()}
          type="button"
          buttonStyle="btn--success--outline"
          buttonSize="btn--medium"
        >
          Predict Handwriting
        </Button>
      );
    }
    return (
      <>
        {drawingCanvas}
        {loading_predict}
        <Button
          onClick={() => this.clearCanvas()}
          type="button"
          buttonStyle="btn--primary--outline"
          buttonSize="btn--medium"
        >
          Clear
        </Button>
      </>
    );
  }
}
