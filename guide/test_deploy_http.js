import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs'
import * as path from 'path'
import axios from 'axios'

let tensor, tensor4D
const MODEL_URL = 'http://167.99.78.252:9000/v1/models/obj_det/versions/1:predict'
const headers = {
    'content-type': 'application/json'
}
const data = {
    'signature_name': 'serving_default',
    'instances': ''
}


// load image and convert to tensor
const img = fs.readFileSync('/root/tensorflow/examples/image1.jpg')

// for running ML model in node. Clear up all tensors after terminated
tf.tidy(() => {
    tensor = tf.node.decodeImage(img) // [800, 1280, 3]
    tensor4D = tensor.expandDims(0) // [1, 800, 1280, 3]
    // console.log(tensor)
    // console.log(tensor4D.dataSync())

    console.log(tensor4D)
    data['instances'] = [...tensor4D.arraySync()]
    axios.post(MODEL_URL, data, headers).then((result) =>
        fs.writeFileSync('./result_tf_js.txt', JSON.stringify(result.data, null, 2), 'utf-8')
    )
})
