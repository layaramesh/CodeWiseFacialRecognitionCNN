using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CSharpFaceRecognition
{
    internal class Program
    {
        // FER+ label order from the ONNX model zoo:
        // [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
        private static readonly string[] Labels = new[]
        {
            "Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"
        };

        static int Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: CSharpFaceRecognition --model <model.onnx> [--image <image.jpg> | --dir <folder>] [--cascade <haarcascade.xml>] [--norm01] [--no-clahe] [--neutral-thresh <0..1>]");
                return 1;
            }

            string modelPath = null!;
            string imagePath = null!;
            string? dirPath = null;
            string? cascadePath = null;
            bool norm01 = false;
            bool noClahe = false;
            bool csvMode = false;
            double neutralThresh = -1; // if set >=0, reassign when Neutral is below threshold

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--model") modelPath = args[++i];
                else if (args[i] == "--image") imagePath = args[++i];
                else if (args[i] == "--dir") dirPath = args[++i];
                else if (args[i] == "--cascade") cascadePath = args[++i];
                else if (args[i] == "--norm01") norm01 = true;
                else if (args[i] == "--no-clahe") noClahe = true;
                else if (args[i] == "--csv") csvMode = true;
                else if (args[i] == "--neutral-thresh" && i + 1 < args.Length)
                {
                    if (!double.TryParse(args[++i], out neutralThresh)) neutralThresh = -1;
                }
            }

            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                Console.Error.WriteLine("Model file not found or not provided.");
                return 2;
            }

            if (string.IsNullOrEmpty(imagePath) && string.IsNullOrEmpty(dirPath))
            {
                Console.Error.WriteLine("Provide either --image <file> or --dir <folder>.");
                return 2;
            }

            try
            {
                using var session = new InferenceSession(modelPath);
                var inputMeta = session.InputMetadata.Values.First();
                var dimsMeta = inputMeta.Dimensions.ToArray();
                bool nhwc = false;
                int channels;
                int height;
                int width;
                if (dimsMeta.Length >= 4)
                {
                    // Detect layout by channel position (NHWC typically has C in last axis)
                    nhwc = dimsMeta[3] == 3 || dimsMeta[3] == 1;
                    channels = (nhwc ? dimsMeta[3] : dimsMeta[1]);
                    height = (nhwc ? dimsMeta[1] : dimsMeta[2]);
                    width = (nhwc ? dimsMeta[2] : dimsMeta[3]);
                    if (channels <= 0) channels = 3; // default
                    if (height <= 0) height = 224;
                    if (width <= 0) width = 224;
                }
                else
                {
                    nhwc = false;
                    channels = 1;
                    height = 64;
                    width = 64;
                }

                if (!string.IsNullOrEmpty(imagePath))
                {
                    return RunSingle(session, imagePath, channels, width, height, nhwc, cascadePath, norm01, noClahe, neutralThresh, csvMode);
                }
                else
                {
                    var exts = new[] { ".jpg", ".jpeg", ".png", ".bmp" };
                    var files = Directory.EnumerateFiles(dirPath!, "*.*", SearchOption.TopDirectoryOnly)
                        .Where(f => exts.Contains(Path.GetExtension(f).ToLowerInvariant()))
                        .OrderBy(f => f)
                        .ToList();
                    if (files.Count == 0)
                    {
                        Console.Error.WriteLine("No images found in directory.");
                        return 3;
                    }

                    int ok = 0, fail = 0;
                    foreach (var f in files)
                    {
                        int code = RunSingle(session, f, channels, width, height, nhwc, cascadePath, norm01, noClahe, neutralThresh, csvMode);
                        if (code == 0) ok++; else fail++;
                    }
                    Console.WriteLine($"Completed. Success: {ok}, Failed: {fail}");
                    return fail == 0 ? 0 : 4;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 10;
            }
        }

        private static int RunSingle(InferenceSession session, string imagePath, int channels, int width, int height, bool nhwc, string? cascadePath, bool norm01, bool noClahe, double neutralThresh, bool csvMode)
        {
            using var img = Cv2.ImRead(imagePath);

            if (img.Empty())
            {
                Console.Error.WriteLine($"Failed to read image: {imagePath}");
                return 3;
            }

            var inputName = session.InputMetadata.Keys.First();
            NamedOnnxValue input;

            if (channels == 1)
            {
                using var grayFull = new Mat();
                Cv2.CvtColor(img, grayFull, ColorConversionCodes.BGR2GRAY);

                Rect roi;
                if (!string.IsNullOrEmpty(cascadePath) && File.Exists(cascadePath))
                {
                    using var cascade = new CascadeClassifier(cascadePath);
                    using var detGray = new Mat();
                    Cv2.EqualizeHist(grayFull, detGray);
                    var faces = cascade.DetectMultiScale(detGray, 1.1, 4, HaarDetectionTypes.ScaleImage, new Size(30, 30));
                    if (faces.Length > 0)
                        roi = faces.OrderByDescending(r => r.Width * r.Height).First();
                    else
                    {
                        int side = Math.Min(img.Width, img.Height);
                        int cx = (img.Width - side) / 2;
                        int cy = (img.Height - side) / 2;
                        roi = new Rect(cx, cy, side, side);
                    }
                }
                else
                {
                    int side = Math.Min(img.Width, img.Height);
                    int cx = (img.Width - side) / 2;
                    int cy = (img.Height - side) / 2;
                    roi = new Rect(cx, cy, side, side);
                }

                using var gray = new Mat(grayFull, roi);
                if (!noClahe)
                {
                    using var clahe = Cv2.CreateCLAHE(2.0, new Size(8, 8));
                    clahe.Apply(gray, gray);
                }
                using var resized = new Mat();
                Cv2.Resize(gray, resized, new Size(width, height));

                var floatData = new float[1 * 1 * width * height];
                for (int row = 0; row < height; row++)
                {
                    for (int col = 0; col < width; col++)
                    {
                        var v = resized.At<byte>(row, col);
                        floatData[row * width + col] = norm01 ? v / 255.0f : v;
                    }
                }

                // Shape per layout
                var shape = nhwc ? new[] { 1, height, width, 1 } : new[] { 1, 1, height, width };
                var tensor = new DenseTensor<float>(floatData, shape);
                input = NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else
            {
                // Assume RGB 3-channel classification (e.g., MobileNet)
                using var resized = new Mat();
                Cv2.Resize(img, resized, new Size(width, height));
                using var rgb = new Mat();
                Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

                // Normalize to ImageNet mean/std if norm01; otherwise keep 0..255 scale
                float[] mean = norm01 ? new[] { 0.485f, 0.456f, 0.406f } : new[] { 0f, 0f, 0f };
                float[] std = norm01 ? new[] { 0.229f, 0.224f, 0.225f } : new[] { 1f, 1f, 1f };

                var floatData = new float[1 * 3 * width * height];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = rgb.At<Vec3b>(y, x);
                        float r = pixel.Item0 / (norm01 ? 255.0f : 1.0f);
                        float g = pixel.Item1 / (norm01 ? 255.0f : 1.0f);
                        float b = pixel.Item2 / (norm01 ? 255.0f : 1.0f);
                        r = (r - mean[0]) / std[0];
                        g = (g - mean[1]) / std[1];
                        b = (b - mean[2]) / std[2];

                        int baseIdx = y * width + x;
                        if (nhwc)
                        {
                            // NHWC: [H, W, C]
                            floatData[baseIdx * 3 + 0] = r;
                            floatData[baseIdx * 3 + 1] = g;
                            floatData[baseIdx * 3 + 2] = b;
                        }
                        else
                        {
                            // NCHW
                            floatData[0 * width * height + baseIdx] = r;
                            floatData[1 * width * height + baseIdx] = g;
                            floatData[2 * width * height + baseIdx] = b;
                        }
                    }
                }

                var shape = nhwc ? new[] { 1, height, width, 3 } : new[] { 1, 3, height, width };
                var tensor = new DenseTensor<float>(floatData, shape);
                input = NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }

            using var results = session.Run(new[] { input });
            var outputTensor = results.First().AsEnumerable<float>().ToArray();

            // Skip face embedding models (high-dimensional outputs like 512)
            if (outputTensor.Length > 100)
            {
                if (!csvMode) Console.WriteLine($"Image: {Path.GetFileName(imagePath)} - Embedding model detected (output dim: {outputTensor.Length}), skipping classification.");
                else Console.WriteLine($"{Path.GetFileName(imagePath)},N/A");
                return 0;
            }

            var probs = Softmax(outputTensor);

            // Detect emotion models by output size (7-8 classes)
            bool isEmotionModel = probs.Length >= 7 && probs.Length <= 8;

            if (csvMode)
            {
                // CSV mode: filename,prediction
                if (isEmotionModel)
                {
                    int best = Array.IndexOf(probs, probs.Max());
                    int neutralIndex = Array.IndexOf(Labels, "Neutral");
                    if (neutralThresh >= 0 && best == neutralIndex && probs[best] < neutralThresh)
                    {
                        int alt = -1;
                        float altVal = float.MinValue;
                        for (int i = 0; i < probs.Length && i < Labels.Length; i++)
                        {
                            if (i == neutralIndex) continue;
                            if (probs[i] > altVal) { altVal = probs[i]; alt = i; }
                        }
                        if (alt >= 0) best = alt;
                    }
                    string label = best >= 0 && best < Labels.Length ? Labels[best] : best.ToString();
                    Console.WriteLine($"{Path.GetFileName(imagePath)},{label}");
                }
                else
                {
                    int best = Array.IndexOf(probs, probs.Max());
                    Console.WriteLine($"{Path.GetFileName(imagePath)},Class {best}");
                }
            }
            else
            {
                Console.WriteLine($"Image: {Path.GetFileName(imagePath)}");

                if (isEmotionModel)
                {
                    int best = Array.IndexOf(probs, probs.Max());
                    int neutralIndex = Array.IndexOf(Labels, "Neutral");
                    if (neutralThresh >= 0 && best == neutralIndex && probs[best] < neutralThresh)
                    {
                        int alt = -1;
                        float altVal = float.MinValue;
                        for (int i = 0; i < probs.Length && i < Labels.Length; i++)
                        {
                            if (i == neutralIndex) continue;
                            if (probs[i] > altVal) { altVal = probs[i]; alt = i; }
                        }
                        if (alt >= 0) best = alt;
                    }
                    string label = best >= 0 && best < Labels.Length ? Labels[best] : best.ToString();
                    Console.WriteLine($"Prediction: {label}");
                    for (int i = 0; i < probs.Length && i < Labels.Length; i++)
                        Console.WriteLine($"  {Labels[i]}: {probs[i]:P2}");
                }
                else
                {
                    // Print Top-5 classes
                    var indices = probs
                        .Select((p, i) => (p, i))
                        .OrderByDescending(t => t.p)
                        .Take(5)
                        .ToArray();
                    for (int k = 0; k < indices.Length; k++)
                    {
                        var (p, i) = indices[k];
                        Console.WriteLine($"  Top {k + 1}: Class {i} ({p:P2})");
                    }
                }
            }

            return 0;
        }

        private static float[] Softmax(float[] logits)
        {
            var max = logits.Max();
            var exps = logits.Select(x => Math.Exp(x - max)).ToArray();
            var sum = exps.Sum();
            return exps.Select(x => (float)(x / sum)).ToArray();
        }
    }
}
