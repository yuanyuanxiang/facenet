#----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rishi Rai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#----------------------------------------------------

import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import align.detect_face

# 初始化图
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# 检测图像数据
def test_src(image_src):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    image_np = np.array(image_src).astype(np.uint8)
    bounding_boxes, _ = align.detect_face.detect_face(image_np, minsize, pnet, rnet, onet, threshold, factor)
    result = np.empty(bounding_boxes.shape, np.float32)
    result[:] = bounding_boxes[:]

    return result

# 检测图像文件
def test_image(image_file):
    try:
        image = Image.open(image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    bounding_boxes = test_src(image)
    print('bounding_boxes.shape =', bounding_boxes.shape)
    print('bounding_boxes.dtype =', bounding_boxes.dtype)
    print('bounding_boxes =', bounding_boxes)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_image('image.jpg')
    else:
        test_image(sys.argv[1])
