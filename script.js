const video = document.getElementById('video')

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}

video.addEventListener('play', async () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  // canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
  const displaySize = { width: video.width, height: video.height }
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  faceapi.matchDimensions(canvas, displaySize)
  
  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    // console.log(resizedDetections)
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
      drawBox.draw(canvas)
    })

    // console.log(results);
    
  }, 200)
})

function loadLabeledImages() {
  const labels = ['Captain America']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      const img = await faceapi.fetchImage(`https://media.gettyimages.com/photos/actor-chris-evans-attends-the-gifted-new-york-premiere-at-new-york-picture-id665462094?k=20&m=665462094&s=612x612&w=0&h=R1q73IWm9ZkpLh0RSECX6f_QapINK6IFKkMp7SMWWII=`)
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
      descriptions.push(detections.descriptor)
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
