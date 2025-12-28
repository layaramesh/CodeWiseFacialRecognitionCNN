# CodeWiseFacialRecognitionCNN
=====================

Improving CodeWise Student Satisfaction using CNN-based Facial Expression Analysis for Enhanced Coaching in and  outside of Classrooms 
CSharpFaceRecognition
This is a minimal C# console application that performs inference with an ONNX model for facial expression recognition.

It uses:
- `Microsoft.ML.OnnxRuntime` for ONNX inference
- `OpenCvSharp4` for image I/O and Haarcascade face detection

Quick start (PowerShell):

```powershell
# From repo root, build the project
cd <path_to_github>\CodeWiseFacialRecognitionCNN
dotnet restore
dotnet build -c Release

# Run (provide an ONNX model and an input image). Optional: provide Haarcascade XML for face detection
dotnet run --project . --model ..\..\model.onnx --image ..\..\test_face.jpg --cascade 
"C:\path\to\haarcascade_frontalface_default.xml"
```
dotnet run --project . --image 'Happy.jpg' --model .\emotion.onnx

Notes:
- The code expects an ONNX model that accepts a single 48x48 grayscale image. 
- Labels are currently the FER2013 labels: `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`.

**Troubleshooting native runtime errors**
- If you see an error like "The type initializer for 'OpenCvSharp.Internal.NativeMethods' threw an exception", it's usually because the native OpenCV runtime DLLs couldn't be loaded. Common fixes:
	- Install the **Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64)**: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
	- Build/run with the Windows runtime identifier so native libraries are included/copied:

```powershell
# build for win-x64 (adds native runtimes)
dotnet build -c Release -r win-x64

# or run with RID
dotnet run --project . -c Release -r win-x64 -- --model .\model.onnx --image ..\test_face.jpg
```

	- Make sure your process architecture (x64 vs x86) matches the native binaries. The project is configured to target `win-x64` by default.
