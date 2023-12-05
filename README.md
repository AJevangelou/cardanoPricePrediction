# Cardano Price Prediction

## Overview

This repository contains code for predicting Cardano (ADA) cryptocurrency prices using LSTM and GRU models. The models are trained on historical price data and evaluated on a test set.

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Background

Trying to explore the viability of using RNN, specifically LSTM and GRU models in predicting highly volatile 
cryptocoin prices such as Cardano

## Dataset

Dataset consists of daily prices of Cardano since 2017. The dataset contains the opening, closing, highest and lowest
price on each date as well as the transaction volume. From these data, the opening price for each day is used in this 
analysis.

## Models

- LSTM
- GRU
- Bidirectional LSTM

## Results

MSE of 0.03582 was accomplished with this analysis.

## Dependencies

This project has implemented Python and PyTorch.

## License

MIT License

