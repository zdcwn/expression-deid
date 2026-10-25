# Third-party expression backbone

The accuracy video backend derives from `enet_b2_7.onnx` published by
[EmotiEffLib / Andrey V. Savchenko and contributors](https://github.com/sb-ai-lab/EmotiEffLib),
at commit `520a051c64cd191521e5934655314e769a319684`.

The upstream repository publishes its code under Apache License 2.0; the
unchanged license is retained in `EmotiEffLib-Apache-2.0.txt`. The downloaded
ONNX graph is converted with [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch)
(version 1.5.15, Apache License 2.0). The converter is a training/export
dependency and is not required by the desktop runtime.

Local changes: discard the published seven-class head; train a new head and
fine tune the last convolution blocks on already anonymous grayscale images;
export the resulting PyTorch graph. The deployed ensemble and calibration
are local project additions. No raw source face is passed to either classifier.

Upstream requested citation:

- A. V. Savchenko, L. V. Savchenko, I. Makarov, "Classifying Emotions and
  Engagement in Online Learning Based on a Single Facial Expression
  Recognition Neural Network", IEEE Transactions on Affective Computing,
  2022. https://ieeexplore.ieee.org/document/9815154
- A. V. Savchenko, "Facial expression recognition with Adaptive Frame Rate
  based on Multiple Testing Correction", ICML 2023.
  https://proceedings.mlr.press/v202/savchenko23a.html

A code license does not replace the original terms for RAF-DB, KMU,
PPB-Emo, AffectNet, or VGGFace2. Their datasets are not included in this
model patch bundle. The local validation scope and asset hashes are recorded
separately; upstream benchmark scores are not used as this project's results.
