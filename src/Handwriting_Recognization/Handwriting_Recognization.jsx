import React, { Component } from "react";
import "./Handwriting_Recognization.css";
import { Button } from "../components/Button.jsx";
import * as tf from "@tensorflow/tfjs";
import { MnistData } from "./data";
import { Icon } from "@iconify/react";
import tensorflowIcon from "@iconify/icons-logos/tensorflow";
import { fabric } from "fabric";
import Chart from "react-apexcharts";

///////////Global Variables///////////
let model;
let data;
let canvas;

//traning constants
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

export default class Handwriting_Recognization extends Component {
  //initiate the bar chart
  constructor() {
    super();
    this.state = {
      trainingComplete: false,
      series: [
        {
          name: "Probability",
          data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
      ],
      options: {
        chart: {
          height: 350,
          type: "bar",
        },
        plotOptions: {
          bar: {
            dataLabels: {
              position: "top", // top, center, bottom
            },
          },
        },
        dataLabels: {
          enabled: true,
          formatter: function (val) {
            return val + "%";
          },
          position: "top",
          offsetY: -20,
          style: {
            fontSize: "12px",
            colors: ["#304758"],
          },
        },

        xaxis: {
          categories: [
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
          ],
          position: "buttom",
          axisBorder: {
            show: true,
          },
          axisTicks: {
            show: false,
          },
          crosshairs: {
            fill: {
              type: "gradient",
              gradient: {
                colorFrom: "#D8E3F0",
                colorTo: "#BED1E6",
                stops: [0, 100],
                opacityFrom: 0.4,
                opacityTo: 0.5,
              },
            },
          },
          tooltip: {
            enabled: true,
          },
        },

        yaxis: {
          axisBorder: {
            show: true,
          },
          axisTicks: {
            show: false,
          },
          labels: {
            show: false,
            formatter: function (val) {
              return val + "%";
            },
          },
        },

        title: {
          //text: "Monthly Inflation in Argentina, 2002",
          floating: true,
          offsetY: 330,
          align: "center",
          style: {
            color: "#000000",
          },
        },
      },
    };
  }

  //create the canvas and train the ML model
  async componentDidMount() {
    canvas = new fabric.Canvas("canvas", {
      backgroundColor: "rgb(0, 0, 0)",
    });
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.width = 11;
    canvas.freeDrawingBrush.color = "rgb(255, 255, 255)";

    this.createModel();
    await this.load();
    await this.train();
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

    let out = await model.predict(tensor);
    out = out.dataSync();
    console.log(out);

    let N = new Array(10);
    //round decimal and convert to percentage
    for (let i = 0; i < 10; i++) {
      out[i] *= 100;
      let n = Math.round(out[i] * 100);
      N[i] = n / 100;
    }

    this.setState({
      series: [
        {
          name: "Probability",
          data: N,
        },
      ],
    });
    console.log(out);
  }

  clearCanvas() {
    canvas.clear();
    canvas.setBackgroundColor("rgb(0, 0, 0)");
    this.setState({
      series: [
        {
          name: "Probability",
          data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
      ],
    });
  }

  render() {
    let loading_predict;
    if (!this.state.trainingComplete) {
      //three dot thing
      loading_predict = (
        <div className="loading">
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
          className="loading"
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
        <div className="container1">
          <canvas
            className="drawingCanvas"
            id="canvas"
            width="300"
            height="300"
          ></canvas>
          <Chart
            className="chart"
            options={this.state.options}
            series={this.state.series}
            type="bar"
            width="500"
          />
        </div>
        <div className="container2">
          <Button
            className="but"
            onClick={() => this.clearCanvas()}
            type="button"
            buttonStyle="btn--primary--outline"
            buttonSize="btn--medium"
          >
            Clear
          </Button>
          {loading_predict}
        </div>
      </>
    );
  }
}
// <Icon className="icon" icon={tensorflowIcon} />
