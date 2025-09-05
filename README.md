## 📚 Introduction

`react-native-vision-camera-face-detector` is a React Native library that integrates with the Vision Camera module to provide face detection functionality. It allows you to easily detect faces in real-time using device's front and back camera.

Is this package usefull to you?

<a href="https://www.buymeacoffee.com/luicfrr" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

Or give it a ⭐ on [GitHub](https://github.com/luicfrr/react-native-vision-camera-face-detector).

## 🏗️ Features

- Real-time face detection using front and back camera
- Adjustable face detection settings
- Optional native side face bounds, contour and landmarks auto scaling
- Can be combined with [Skia Frame Processor](https://react-native-vision-camera.com/docs/guides/skia-frame-processors)

## 🧰 Installation

```bash
yarn add react-native-vision-camera-face-detector
```

Then you need to add `react-native-worklets-core` plugin to `babel.config.js`. More details [here](https://react-native-vision-camera.com/docs/guides/frame-processors#react-native-worklets-core).

## 🪲 Knowing Bugs

There are open issues ([here](https://github.com/mrousavy/react-native-vision-camera/issues/3362), [here](https://github.com/mrousavy/react-native-vision-camera/issues/3034), and [here](https://github.com/mrousavy/react-native-vision-camera/issues/2951)) about a bug on Skia Frame Processor that may cause a Black Screen on some Android Devices.
This bug can be easily fixed with [this trick](https://github.com/mrousavy/react-native-vision-camera/issues/3362#issuecomment-2624299305) but it makes Frame drawings to be in incorrect orientation.

## 💡 Usage

Recommended way (see [Example App](https://github.com/luicfrr/react-native-vision-camera-face-detector/blob/main/example/src/index.tsx) for Skia usage):
```jsx
import { 
  StyleSheet, 
  Text, 
  View 
} from 'react-native'
import { 
  useEffect, 
  useState,
  useRef
} from 'react'
import {
  Frame,
  useCameraDevice
} from 'react-native-vision-camera'
import {
  Face,
  Camera,
  FaceDetectionOptions
} from 'react-native-vision-camera-face-detector'

export default function App() {
  const faceDetectionOptions = useRef<FaceDetectionOptions>( {
    // detection options
  } ).current

  const device = useCameraDevice('front')

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission()
      console.log({ status })
    })()
  }, [device])

  function handleFacesDetection(
    faces: Face[],
    frame: Frame
  ) { 
    console.log(
      'faces', faces.length,
      'frame', frame.toString()
    )
  }

  return (
    <View style={{ flex: 1 }}>
      {!!device? <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        faceDetectionCallback={ handleFacesDetection }
        faceDetectionOptions={ faceDetectionOptions }
      /> : <Text>
        No Device
      </Text>}
    </View>
  )
}
```

Or use it following [vision-camera docs](https://react-native-vision-camera.com/docs/guides/frame-processors-interacting):
```jsx
import { 
  StyleSheet, 
  Text, 
  View,
  NativeModules,
  Platform
} from 'react-native'
import { 
  useEffect, 
  useState,
  useRef
} from 'react'
import {
  Camera,
  runAsync,
  useCameraDevice,
  useFrameProcessor
} from 'react-native-vision-camera'
import { 
  Face,
  useFaceDetector,
  FaceDetectionOptions
} from 'react-native-vision-camera-face-detector'
import { Worklets } from 'react-native-worklets-core'

export default function App() {
  const faceDetectionOptions = useRef<FaceDetectionOptions>( {
    // detection options
  } ).current

  const device = useCameraDevice('front')
  const { 
    detectFaces,
    stopListeners
  } = useFaceDetector( faceDetectionOptions )

  useEffect( () => {
    return () => {
      // you must call `stopListeners` when current component is unmounted
      stopListeners()
    }
  }, [] )

  useEffect(() => {
    if(!device) {
      // you must call `stopListeners` when `Camera` component is unmounted
      stopListeners()
      return
    }

    (async () => {
      const status = await Camera.requestCameraPermission()
      console.log({ status })
    })()
  }, [device])

  const handleDetectedFaces = Worklets.createRunOnJS( (
    faces: Face[]
  ) => { 
    console.log( 'faces detected', faces )
  })

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    runAsync(frame, () => {
      'worklet'
      const faces = detectFaces(frame)
      // ... chain some asynchronous frame processor
      // ... do something asynchronously with frame
      handleDetectedFaces(faces)
    })
    // ... chain frame processors
    // ... do something with frame
  }, [handleDetectedFaces])

  return (
    <View style={{ flex: 1 }}>
      {!!device? <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
      /> : <Text>
        No Device
      </Text>}
    </View>
  )
}
```

As face detection is a heavy process you should run it in an asynchronous thread so it can be finished without blocking your camera preview.
You should read `vision-camera` [docs](https://react-native-vision-camera.com/docs/guides/frame-processors-interacting#running-asynchronously) about this feature.

## 🖼️ Static Image Face Detection

You can detect whether a static image contains a face without using the camera.

- Returns `true` if at least one face is detected.
- Accepts `require('path/to/file')`, a URI string (e.g. `file://`, `content://`, `http(s)://`), or an object `{ uri: string }`.

```ts
import { hasFace } from 'react-native-vision-camera-face-detector'

// Using a bundled asset
const result1 = await hasFace(require('./assets/photo.jpg')) // true | false

// Using a local file path or content URI (e.g. from an image picker)
const result2 = await hasFace('file:///storage/emulated/0/Download/pic.jpg')
const result3 = await hasFace({ uri: 'content://media/external/images/media/12345' })

console.log({ result1, result2, result3 })
```

To get the number of faces in an image:

```ts
import { countFaces } from 'react-native-vision-camera-face-detector'
const n = await countFaces(require('./assets/group.jpg')) // e.g. 3
```


## Face Detection Options

| Option  | Description | Default | Options |
| ------------- | ------------- | ------------- | ------------- |
| `cameraFacing` | Current active camera | `front` | `front`, `back` |
| `performanceMode` | Favor speed or accuracy when detecting faces.  | `fast` | `fast`, `accurate`|
| `landmarkMode` | Whether to attempt to identify facial `landmarks`: eyes, ears, nose, cheeks, mouth, and so on. | `none` | `none`, `all` |
| `contourMode` | Whether to detect the contours of facial features. Contours are detected for only the most prominent face in an image. | `none` | `none`, `all` |
| `classificationMode` | Whether or not to classify faces into categories such as 'smiling', and 'eyes open'. | `none` | `none`, `all` |
| `minFaceSize` | Sets the smallest desired face size, expressed as the ratio of the width of the head to width of the image. | `0.15` | `number` |
| `trackingEnabled` | Whether or not to assign faces an ID, which can be used to track faces across images. Note that when contour detection is enabled, only one face is detected, so face tracking doesn't produce useful results. For this reason, and to improve detection speed, don't enable both contour detection and face tracking. | `false` | `boolean` |
| `autoMode` | Should handle auto scale (face bounds, contour and landmarks) and rotation on native side? If this option is disabled all detection results will be relative to frame coordinates, not to screen/preview. You shouldn't use this option if you want to draw on screen using `Skia Frame Processor`. See [this](https://github.com/luicfrr/react-native-vision-camera-face-detector/issues/30#issuecomment-2058805546) and [this](https://github.com/luicfrr/react-native-vision-camera-face-detector/issues/35) for more details. | `false` | `boolean` |
| `windowWidth` | * Required if you want to use `autoMode`. You must handle your own logic to get screen sizes, with or without statusbar size, etc... | `1.0` | `number` |
| `windowHeight` | * Required if you want to use `autoMode`. You must handle your own logic to get screen sizes, with or without statusbar size, etc... | `1.0` | `number` |

## 🔧 Troubleshooting

Here is a common issue when trying to use this package and how you can try to fix it:

- `Regular javascript function cannot be shared. Try decorating the function with the 'worklet' keyword...`:
  - If you're using `react-native-reanimated` maybe you're missing [this](https://github.com/mrousavy/react-native-vision-camera/issues/1791#issuecomment-1892130378) step.
- `Execution failed for task ':react-native-vision-camera-face-detector:compileDebugKotlin'...`:
  - This error is probably related to gradle cache. Try [this](https://github.com/luicfrr/react-native-vision-camera-face-detector/issues/71#issuecomment-2186614831) sollution first.
  - Also check [this](https://github.com/luicfrr/react-native-vision-camera-face-detector/issues/90#issuecomment-2358160166) comment.

If you find other errors while using this package you're wellcome to open a new issue or create a PR with the fix.

## 👷 Built With

- [React Native](https://reactnative.dev/)
- [Google MLKit](https://developers.google.com/ml-kit)
- [Vision Camera](https://react-native-vision-camera.com/)

## 🔎 About

This package was tested using the following:

- `react-native`: `0.76.9` (new arch disabled)
- `react-native-vision-camera`: `4.6.4`
- `react-native-worklets-core`: `1.5.0`
- `@shopify/react-native-skia`: `1.5.0`
- `react-native-reanimated`: `~3.16.1`
- `@react-native-firebase`: `^22.2.0`
- `expo`: `^52`

Min O.S version:

- `Android`: `SDK 26` (Android 8)
- `IOS`: `15.5`

Make sure to follow tested versions and your device is using the minimum O.S version before opening issues.

## 📚 Author

Made with ❤️ by [luicfrr](https://github.com/luicfrr)
