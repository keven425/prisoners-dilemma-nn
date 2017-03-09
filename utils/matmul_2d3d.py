import tensorflow as tf
import numpy as np

# matrix multiplication of 3D shape (?, i, j) by 2D shape (k, l)

def m_3d2d(a, b):
  # a: 3D
  # b: 2D
  input_size = a.get_shape()[-1].value
  max_timestep = a.get_shape()[1].value
  output_size = b.get_shape()[-1].value

  a = tf.reshape(a, [-1, input_size])
  prod = tf.matmul(a, b)
  prod = tf.reshape(prod, [-1, max_timestep, output_size])
  return prod


def m_2d3d(a, b):
  # a: 2D
  # b: 3D
  input_size = a.get_shape()[-1].value
  output_size1 = b.get_shape()[1].value
  output_size2 = b.get_shape()[-1].value

  b = tf.reshape(a, [input_size, -1])
  prod = tf.matmul(a, b)
  prod = tf.reshape(prod, [-1, output_size1, output_size2])
  return prod



def test_matmul_3d2d():
  print('experimenting w/ matmul 3d x 2d')

  with tf.variable_scope("test_matmul"):
    m1_placeholder = tf.placeholder(tf.float32, shape=(None, 2, 3))
    m2_placeholder = tf.placeholder(tf.float32, shape=(3, 2))

  init = tf.global_variables_initializer()

  with tf.Session() as session:
    session.run(init)
    input1 = np.array([
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]
    ], dtype=np.float32)
    input2 = np.array([
      [1, 2], [3, 4], [5, 6]
    ], dtype=np.float32)

    prod = m_3d2d(m1_placeholder, m2_placeholder)

    """
    [[[  22.   28.]
    [  49.   64.]]

    [[  76.  100.]
    [ 103.  136.]]]
    """

    _prod = session.run(prod, feed_dict={m1_placeholder: input1, m2_placeholder: input2})
    print("prod = " + str(_prod))


def test_matmul_2d3d():
  print('experimenting w/ matmul 2d x 3d')

  with tf.variable_scope("test_matmul"):
    m1_placeholder = tf.placeholder(tf.float32, shape=(None, 2))
    m2_placeholder = tf.placeholder(tf.float32, shape=(2, 3, 2))

  init = tf.global_variables_initializer()

  with tf.Session() as session:
    session.run(init)
    input1 = np.array([
      [1, 2], [3, 4], [5, 6]
    ], dtype=np.float32)
    input2 = np.array([
      [1, 2], [3, 4], [5, 6]
    ], dtype=np.float32)

    prod = m_2d3d(m1_placeholder, m2_placeholder)

    """
    [[[  22.   28.]
    [  49.   64.]]

    [[  76.  100.]
    [ 103.  136.]]]
    """

    _prod = session.run(prod, feed_dict={m1_placeholder: input1, m2_placeholder: input2})
    print("prod = " + str(_prod))


if __name__ == "__main__":
  test_matmul()


